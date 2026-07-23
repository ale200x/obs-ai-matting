// obs-ai-matting — OBS filter for AI background matting (RVM via ONNX Runtime / GPU).
// Copyright (C) 2026 ale200x
//
// This program is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License version 2 as published by the Free Software
// Foundation. See the LICENSE file. Distributed WITHOUT ANY WARRANTY.
//
// Inferencia en hilo aparte (baja latencia). Modos: transparente o blur.
#include <obs-module.h>
#include <graphics/graphics.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <cstdlib>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-ai-matting", "en-US")

static bool file_exists(const std::string &p)
{
	std::ifstream f(p);
	return f.good();
}

// Resuelve la ruta del modelo: ajuste del filtro > $OBS_AI_MATTING_MODEL >
// carpeta de datos del plugin > rutas comunes.
static std::string resolve_model_path(obs_data_t *s)
{
	const char *set = s ? obs_data_get_string(s, "model_path") : nullptr;
	if (set && *set && file_exists(set)) return set;
	if (const char *env = getenv("OBS_AI_MATTING_MODEL"))
		if (*env && file_exists(env)) return env;
	const char *home = getenv("HOME");
	std::string h = home ? home : "";
	std::vector<std::string> cand = {
		h + "/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx",
		h + "/ai-camera/models/rvm_resnet50.onnx",
	};
	for (auto &c : cand) if (file_exists(c)) return c;
	return cand.front();  // por defecto (aunque no exista aún)
}

struct am_filter {
	obs_source_t *context = nullptr;

	gs_texrender_t *texrender = nullptr;
	gs_stagesurf_t *stage = nullptr;
	gs_texture_t *out_tex = nullptr;
	uint32_t width = 0, height = 0;

	// settings
	std::atomic<int> mode{0};            // 0 transparente, 1 blur
	std::atomic<double> blur{22.0};
	std::atomic<double> alpha_gamma{0.75};
	std::atomic<int> matte_long{512};
	std::atomic<int> detail_ratio{75};   // downsample_ratio de RVM en porcentaje
	uint8_t lut[256];                    // brillo+gamma (UI -> render)
	std::string model_path;

	// onnx (solo el worker lo toca)
	Ort::Env *env = nullptr;
	Ort::Session *session = nullptr;
	Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<Ort::Value> rec;
	int rec_w = 0, rec_h = 0, rec_ratio = 0;
	bool ort_ok = false;

	// buffers render-thread
	std::vector<uint8_t> bgra;           // frame staged+brillo (w*h*4)

	// handoff render -> worker
	std::thread worker;
	std::mutex in_mtx;
	std::condition_variable in_cv;
	std::vector<uint8_t> in_bgra;
	int in_w = 0, in_h = 0;
	bool in_new = false;
	std::atomic<bool> stop{false};

	// salida worker -> render
	std::mutex out_mtx;
	std::vector<float> out_alpha;
	std::vector<uint8_t> out_bgra;       // mismo frame usado para calcular out_alpha
	int out_w = 0, out_h = 0;
	float out_fg_mean[3] = {0, 0, 0};    // media BGR 0..1 del sujeto (pond. por alfa)
	bool out_fg_valid = false;

	// --- igualar luz con el fondo ---
	std::atomic<bool> auto_match{false};
	std::atomic<double> match_strength{0.7};
	std::mutex bg_mtx;                   // protege bg_name + bg_weak (UI <-> render)
	std::string bg_name;
	obs_weak_source_t *bg_weak = nullptr;
	int bg_retry = 0;                    // countdown de reintentos de resolucion
	gs_texrender_t *bg_texrender = nullptr;
	gs_stagesurf_t *bg_stage = nullptr;  // BG_W x BG_H fijo
	bool rendering_bg = false;           // guard de reentrancia (solo hilo grafico)
	uint32_t frame_idx = 0;
	int bg_miss = 0;                     // muestreos de fondo fallidos consecutivos
	// estado del match (solo render thread)
	float bg_mean[3] = {0, 0, 0};        // media BGR 0..1 del fondo
	bool bg_valid = false;
	float auto_gain[3] = {1, 1, 1};      // ganancias suavizadas (EMA)
	float baked_gain[3] = {1, 1, 1};     // ultimo valor horneado en auto_lut
	uint8_t auto_lut[3][256];            // gain por canal B/G/R (identidad si off)
};

static constexpr int BG_W = 64, BG_H = 36;   // tamano del muestreo del fondo

static void rebuild_lut(am_filter *f, double gain, double gamma)
{
	for (int i = 0; i < 256; i++) {
		double v = i / 255.0;
		if (gamma != 1.0) v = pow(v, gamma);
		v *= gain;
		f->lut[i] = (uint8_t)std::clamp(v * 255.0, 0.0, 255.0);
	}
}

static void reset_states(am_filter *f)
{
	static float zero = 0.f;
	f->rec.clear();
	for (int i = 0; i < 4; i++) {
		int64_t shp[4] = {1, 1, 1, 1};
		f->rec.push_back(Ort::Value::CreateTensor<float>(f->mem, &zero, 1, shp, 4));
	}
	f->rec_w = 0;
	f->rec_h = 0;
	f->rec_ratio = 0;
}

static void ort_init(am_filter *f)
{
	delete f->session; f->session = nullptr;
	delete f->env; f->env = nullptr;
	f->ort_ok = false;
	if (!file_exists(f->model_path)) {
		blog(LOG_ERROR, "[obs-ai-matting] modelo no encontrado: %s "
			"(descarga rvm_resnet50.onnx, ver README)", f->model_path.c_str());
		return;
	}
	try {
		f->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "obs-ai-matting");
		Ort::SessionOptions so;
		so.SetIntraOpNumThreads(1);
		bool cuda = false;
		for (auto &p : Ort::GetAvailableProviders())
			if (p == "CUDAExecutionProvider") cuda = true;
		if (cuda) {
			OrtCUDAProviderOptions cu{};
			cu.device_id = 0;
			so.AppendExecutionProvider_CUDA(cu);
			blog(LOG_INFO, "[obs-ai-matting] CUDA ON");
		} else {
			blog(LOG_WARNING, "[obs-ai-matting] sin CUDA -> CPU");
		}
		f->session = new Ort::Session(*f->env, f->model_path.c_str(), so);
		reset_states(f);
		f->ort_ok = true;
		blog(LOG_INFO, "[obs-ai-matting] modelo: %s", f->model_path.c_str());
	} catch (const std::exception &e) {
		blog(LOG_ERROR, "[obs-ai-matting] onnx init: %s", e.what());
		f->ort_ok = false;
	}
}

static void resize_alpha(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / dh - 0.5f;
		int y0 = (int)floorf(fy); float wy = fy - y0;
		int y0c = std::clamp(y0, 0, sh - 1), y1c = std::clamp(y0 + 1, 0, sh - 1);
		for (int x = 0; x < dw; x++) {
			float fx = (x + 0.5f) * sw / dw - 0.5f;
			int x0 = (int)floorf(fx); float wx = fx - x0;
			int x0c = std::clamp(x0, 0, sw - 1), x1c = std::clamp(x0 + 1, 0, sw - 1);
			float a = src[y0c*sw+x0c], b = src[y0c*sw+x1c];
			float c = src[y1c*sw+x0c], d = src[y1c*sw+x1c];
			dst[y*dw+x] = (a*(1-wx)+b*wx)*(1-wy) + (c*(1-wx)+d*wx)*wy;
		}
	}
}

// corre RVM sobre buf BGRA(W,H) -> out (W*H alpha) + media BGR del sujeto
// ponderada por alfa (para igualar luz con el fondo). Solo el worker llama esto.
static void run_matting(am_filter *f, const uint8_t *buf, int W, int H, std::vector<float> &out,
			float fg_mean[3], float *fg_weight)
{
	int ml = f->matte_long.load();
	float sc = (float)ml / std::max(W, H);
	int mw = std::max(16, ((int)lround(W * sc) / 16) * 16);
	int mh = std::max(16, ((int)lround(H * sc) / 16) * 16);
	int ratio_pct = std::clamp(f->detail_ratio.load(), 40, 100);
	// Los estados recurrentes de RVM dependen del tamano espacial. Si cambia la
	// fuente, la calidad o el detalle, los estados anteriores no son compatibles.
	if (f->rec_w != mw || f->rec_h != mh || f->rec_ratio != ratio_pct) {
		reset_states(f);
		f->rec_w = mw;
		f->rec_h = mh;
		f->rec_ratio = ratio_pct;
	}

	std::vector<float> src((size_t)3 * mw * mh);
	for (int y = 0; y < mh; y++) {
		int sy = std::min(H - 1, (int)((y + 0.5f) * H / mh));
		for (int x = 0; x < mw; x++) {
			int sx = std::min(W - 1, (int)((x + 0.5f) * W / mw));
			const uint8_t *p = &buf[((size_t)sy * W + sx) * 4];
			src[(0*mh+y)*mw+x] = p[2] / 255.f;
			src[(1*mh+y)*mw+x] = p[1] / 255.f;
			src[(2*mh+y)*mw+x] = p[0] / 255.f;
		}
	}
	int64_t src_shape[4] = {1, 3, mh, mw};
	float dsr = ratio_pct / 100.0f; int64_t dsr_shape[1] = {1};
	const char *in_names[6] = {"src","r1i","r2i","r3i","r4i","downsample_ratio"};
	const char *out_names[5] = {"pha","r1o","r2o","r3o","r4o"};

	std::vector<Ort::Value> ins;
	ins.push_back(Ort::Value::CreateTensor<float>(f->mem, src.data(), src.size(), src_shape, 4));
	for (int i = 0; i < 4; i++) ins.push_back(std::move(f->rec[i]));
	ins.push_back(Ort::Value::CreateTensor<float>(f->mem, &dsr, 1, dsr_shape, 1));

	try {
		auto outs = f->session->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), 6, out_names, 5);
		auto shp = outs[0].GetTensorTypeAndShapeInfo().GetShape();
		int ph = (int)shp[2], pw = (int)shp[3];
		const float *pha = outs[0].GetTensorData<float>();
		double ag = f->alpha_gamma.load();
		std::vector<float> small((size_t)pw * ph);
		for (size_t i = 0; i < small.size(); i++) {
			float a = std::clamp(pha[i], 0.f, 1.f);
			if (ag != 1.0) a = powf(a, (float)ag);
			small[i] = a;
		}
		// media BGR del sujeto sobre los buffers pequenos ya existentes (barato).
		// Alfa continuo como peso: mas estable que un umbral duro con el pelo.
		if (pw == mw && ph == mh) {
			double sr = 0, sg = 0, sb = 0, sa = 0;
			const size_t plane = (size_t)mw * mh;
			for (size_t i = 0; i < small.size(); i++) {
				float a = small[i];
				sa += a;
				sr += src[i] * a;              // plano 0 = R
				sg += src[plane + i] * a;      // plano 1 = G
				sb += src[2 * plane + i] * a;  // plano 2 = B
			}
			if (sa > 0) {
				fg_mean[0] = (float)(sb / sa);
				fg_mean[1] = (float)(sg / sa);
				fg_mean[2] = (float)(sr / sa);
			}
			*fg_weight = (float)(sa / (double)plane);
		}
		out.resize((size_t)W * H);
		resize_alpha(small.data(), pw, ph, out.data(), W, H);
		f->rec.clear();
		for (int i = 1; i <= 4; i++) f->rec.push_back(std::move(outs[i]));
	} catch (const std::exception &e) {
		blog(LOG_ERROR, "[obs-ai-matting] run: %s", e.what());
		reset_states(f);
		out.clear();
	}
}

static void worker_loop(am_filter *f)
{
	std::vector<uint8_t> local; int W, H;
	while (!f->stop.load()) {
		{
			std::unique_lock<std::mutex> lk(f->in_mtx);
			f->in_cv.wait(lk, [&]{ return f->in_new || f->stop.load(); });
			if (f->stop.load()) break;
			local = f->in_bgra; W = f->in_w; H = f->in_h;
			f->in_new = false;
		}
		std::vector<float> alpha;
		float fg_mean[3] = {0, 0, 0}, fg_weight = 0;
		run_matting(f, local.data(), W, H, alpha, fg_mean, &fg_weight);
		if (!alpha.empty()) {
			std::lock_guard<std::mutex> lk(f->out_mtx);
			f->out_alpha = std::move(alpha);
			f->out_bgra = std::move(local);
			f->out_w = W; f->out_h = H;
			for (int c = 0; c < 3; c++) f->out_fg_mean[c] = fg_mean[c];
			f->out_fg_valid = fg_weight > 0.02f;  // <2% de sujeto -> invalido
		}
	}
}

static void box_blur_bgra(const uint8_t *in, uint8_t *out, int W, int H, int radius);

static void stop_worker(am_filter *f)
{
	if (f->worker.joinable()) {
		f->stop = true;
		f->in_cv.notify_all();
		f->worker.join();
	}
	f->stop = false;
}

static void start_worker(am_filter *f)
{
	f->stop = false;
	f->worker = std::thread(worker_loop, f);
}

// ---------- igualar luz con el fondo ----------

// bg_mtx debe estar tomado. Suelta la referencia debil y balancea el inc_showing.
static void release_bg_source(am_filter *f)
{
	if (f->bg_weak) {
		obs_source_t *s = obs_weak_source_get_source(f->bg_weak);
		if (s) { obs_source_dec_showing(s); obs_source_release(s); }
		obs_weak_source_release(f->bg_weak);
		f->bg_weak = nullptr;
	}
}

// Devuelve referencia fuerte a la fuente de fondo (el caller libera) o nullptr.
// Resolucion perezosa por nombre con reintentos: cubre el orden de carga de OBS
// (la fuente puede no existir aun al crear el filtro) y borrado/recreado en vivo.
static obs_source_t *acquire_bg_source(am_filter *f)
{
	std::lock_guard<std::mutex> lk(f->bg_mtx);
	if (f->bg_name.empty()) return nullptr;
	if (f->bg_weak) {
		obs_source_t *s = obs_weak_source_get_source(f->bg_weak);
		if (s) return s;
		obs_weak_source_release(f->bg_weak);  // la fuente murio
		f->bg_weak = nullptr;
		blog(LOG_INFO, "[obs-ai-matting] fuente de fondo perdida: %s", f->bg_name.c_str());
	}
	if (f->bg_retry-- > 0) return nullptr;  // no martillar la busqueda por nombre
	f->bg_retry = 4;                        // ~1 s (se llama cada 15 frames)
	obs_source_t *s = obs_get_source_by_name(f->bg_name.c_str());
	if (!s) return nullptr;
	obs_source_inc_showing(s);  // que actualice video aunque no este visible
	f->bg_weak = obs_source_get_weak_source(s);
	blog(LOG_INFO, "[obs-ai-matting] fuente de fondo enganchada: %s", f->bg_name.c_str());
	return s;
}

// Renderiza la fuente de fondo a BG_W x BG_H (downscale en GPU) y saca su media
// BGR ponderada por alfa. Corre en el hilo grafico, cada pocos frames.
static void sample_background(am_filter *f)
{
	obs_source_t *bg = acquire_bg_source(f);
	if (!bg) {
		if (++f->bg_miss >= 3) f->bg_valid = false;
		return;
	}
	// defensa extra: nunca la propia cadena del filtro
	if (bg == obs_filter_get_parent(f->context) || bg == obs_filter_get_target(f->context)) {
		obs_source_release(bg);
		f->bg_valid = false;
		return;
	}
	uint32_t sw = obs_source_get_width(bg), sh = obs_source_get_height(bg);
	if (!sw || !sh) { obs_source_release(bg); return; }  // conserva la ultima media

	// Si el fondo es una escena que contiene esta camara, OBS reentra en
	// am_render; el guard rendering_bg hace que la camara no aporte nada ->
	// la media es exactamente "la escena sin el sujeto".
	f->rendering_bg = true;
	gs_texrender_reset(f->bg_texrender);
	if (gs_texrender_begin(f->bg_texrender, BG_W, BG_H)) {
		struct vec4 clr; vec4_zero(&clr);
		gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
		gs_ortho(0.0f, (float)sw, 0.0f, (float)sh, -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		obs_source_video_render(bg);
		gs_blend_state_pop();
		gs_texrender_end(f->bg_texrender);
	}
	f->rendering_bg = false;
	obs_source_release(bg);

	gs_stage_texture(f->bg_stage, gs_texrender_get_texture(f->bg_texrender));
	uint8_t *map = nullptr; uint32_t ls = 0;
	if (!gs_stagesurface_map(f->bg_stage, &map, &ls)) return;
	double sum[3] = {0, 0, 0}, suma = 0;
	for (int y = 0; y < BG_H; y++) {
		const uint8_t *row = map + (size_t)y * ls;
		for (int x = 0; x < BG_W; x++) {
			sum[0] += row[x*4+0] / 255.0;
			sum[1] += row[x*4+1] / 255.0;
			sum[2] += row[x*4+2] / 255.0;
			suma   += row[x*4+3] / 255.0;
		}
	}
	gs_stagesurface_unmap(f->bg_stage);
	if (suma < 0.02 * BG_W * BG_H) { f->bg_valid = false; return; }  // fondo vacio
	for (int c = 0; c < 3; c++)
		f->bg_mean[c] = (float)(sum[c] / suma);  // exacto para premultiplicado
	f->bg_valid = true;
	f->bg_miss = 0;
	if ((f->frame_idx % 300) == 0)
		blog(LOG_DEBUG, "[obs-ai-matting] bg_mean B=%.3f G=%.3f R=%.3f",
		     f->bg_mean[0], f->bg_mean[1], f->bg_mean[2]);
}

// Calcula las ganancias objetivo (exposicion parcial + balance de blancos sutil),
// las suaviza con EMA y hornea las LUTs por canal solo si cambiaron.
static void update_auto_gains(am_filter *f, const float fg_mean[3], bool fg_valid)
{
	float target[3] = {1.f, 1.f, 1.f};
	if (f->auto_match.load() && f->mode.load() == 0 && f->bg_valid && fg_valid) {
		auto luma = [](const float m[3]) {  // Rec.709 sobre BGR, gamma-space
			return std::max(0.0722f*m[0] + 0.7152f*m[1] + 0.2126f*m[2], 0.02f);
		};
		float Yf = luma(fg_mean), Yb = luma(f->bg_mean);
		float st = (float)f->match_strength.load();
		// exposicion PARCIAL (k=0.55): el sujeto queda algo mas iluminado que
		// el fondo (convencion de retrato); clamp estrecho para que un fondo
		// negro no hunda al sujeto hasta lo ilegible.
		float e = std::clamp(powf(Yb / Yf, 0.55f), 0.70f, 1.40f);
		for (int c = 0; c < 3; c++) {
			// balance de blancos: acercar la cromaticidad; el clamp hace
			// que un fondo de color saturado insinue, no tina.
			float r = std::clamp((f->bg_mean[c] / Yb) /
					     std::max(fg_mean[c] / Yf, 0.02f), 0.75f, 1.33f);
			target[c] = std::clamp(1.f + st * (e * r - 1.f), 0.5f, 2.f);
		}
	}
	// si esta apagado / falta info, target = identidad -> decaimiento suave
	bool dirty = false;
	for (int c = 0; c < 3; c++) {
		f->auto_gain[c] += 0.04f * (target[c] - f->auto_gain[c]);  // ~0.5-1 s
		if (fabsf(f->auto_gain[c] - f->baked_gain[c]) > 0.002f) dirty = true;
	}
	if (!dirty) return;  // LUTs congeladas en reposo -> cero shimmer
	for (int c = 0; c < 3; c++) {
		f->baked_gain[c] = f->auto_gain[c];
		for (int i = 0; i < 256; i++)
			f->auto_lut[c][i] = (uint8_t)std::clamp(lround(i * f->auto_gain[c]), 0L, 255L);
	}
	if ((f->frame_idx % 300) == 0)
		blog(LOG_DEBUG, "[obs-ai-matting] auto_gain B=%.3f G=%.3f R=%.3f",
		     f->auto_gain[0], f->auto_gain[1], f->auto_gain[2]);
}

// ---------- OBS ----------
static const char *am_get_name(void *) { return obs_module_text("FilterName"); }

static void am_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<am_filter *>(data);
	f->mode = (int)obs_data_get_int(s, "mode");
	f->blur = obs_data_get_double(s, "blur");
	f->alpha_gamma = obs_data_get_double(s, "alpha_gamma");
	f->matte_long = (int)obs_data_get_int(s, "quality");
	f->detail_ratio = (int)obs_data_get_int(s, "detail_ratio");
	rebuild_lut(f, obs_data_get_double(s, "gain"), obs_data_get_double(s, "gamma"));

	f->auto_match = obs_data_get_bool(s, "auto_match");
	f->match_strength = obs_data_get_double(s, "match_strength");
	const char *bn = obs_data_get_string(s, "bg_source");
	{
		std::lock_guard<std::mutex> lk(f->bg_mtx);
		if (f->bg_name != (bn ? bn : "")) {
			release_bg_source(f);
			f->bg_name = bn ? bn : "";
			f->bg_retry = 0;  // resolver ya en el proximo muestreo
		}
	}

	// recarga el modelo si cambió la ruta (también es la carga inicial)
	std::string np = resolve_model_path(s);
	if (np != f->model_path) {
		stop_worker(f);
		f->model_path = np;
		ort_init(f);
		if (f->ort_ok) start_worker(f);
	}
}

static void *am_create(obs_data_t *s, obs_source_t *src)
{
	auto *f = new am_filter();
	f->context = src;
	for (int i = 0; i < 256; i++) f->lut[i] = (uint8_t)i;
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++) f->auto_lut[c][i] = (uint8_t)i;
	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	f->bg_texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	f->bg_stage = gs_stagesurface_create(BG_W, BG_H, GS_BGRA);
	obs_leave_graphics();
	am_update(f, s);   // carga modelo + arranca worker
	return f;
}

static void am_destroy(void *data)
{
	auto *f = static_cast<am_filter *>(data);
	f->stop = true;
	f->in_cv.notify_all();
	if (f->worker.joinable()) f->worker.join();
	{
		std::lock_guard<std::mutex> lk(f->bg_mtx);
		release_bg_source(f);
	}
	obs_enter_graphics();
	if (f->texrender) gs_texrender_destroy(f->texrender);
	if (f->bg_texrender) gs_texrender_destroy(f->bg_texrender);
	if (f->stage) gs_stagesurface_destroy(f->stage);
	if (f->bg_stage) gs_stagesurface_destroy(f->bg_stage);
	if (f->out_tex) gs_texture_destroy(f->out_tex);
	obs_leave_graphics();
	delete f->session;
	delete f->env;
	delete f;
}

static void am_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<am_filter *>(data);
	// Reentrado via escena-como-fondo (sample_background renderiza una escena
	// que contiene esta camara): no aportar nada, asi el muestreo del fondo
	// mide exactamente "la escena sin el sujeto".
	if (f->rendering_bg) return;
	obs_source_t *target = obs_filter_get_target(f->context);
	if (!target) { obs_source_skip_video_filter(f->context); return; }
	uint32_t w = obs_source_get_base_width(target);
	uint32_t h = obs_source_get_base_height(target);
	if (!w || !h || !f->ort_ok) { obs_source_skip_video_filter(f->context); return; }

	if (w != f->width || h != f->height || !f->stage) {
		f->width = w; f->height = h;
		if (f->stage) gs_stagesurface_destroy(f->stage);
		f->stage = gs_stagesurface_create(w, h, GS_BGRA);
		if (f->out_tex) gs_texture_destroy(f->out_tex);
		f->out_tex = gs_texture_create(w, h, GS_BGRA, 1, nullptr, GS_DYNAMIC);
		f->bgra.assign((size_t)w * h * 4, 0);
	}

	// 0) muestrea el fondo elegido (igualar luz); antes del texrender principal
	// para no anidar texrenders
	f->frame_idx++;
	if (f->auto_match.load() && f->mode.load() == 0 && (f->frame_idx % 15) == 0)
		sample_background(f);

	// 1) render input -> texrender
	gs_texrender_reset(f->texrender);
	if (!gs_texrender_begin(f->texrender, w, h)) { obs_source_skip_video_filter(f->context); return; }
	struct vec4 clr; vec4_zero(&clr);
	gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(f->texrender);

	// 2) stage -> CPU BGRA + aplica brillo (LUT)
	gs_stage_texture(f->stage, gs_texrender_get_texture(f->texrender));
	uint8_t *map = nullptr; uint32_t linesize = 0;
	if (gs_stagesurface_map(f->stage, &map, &linesize)) {
		for (uint32_t y = 0; y < h; y++) {
			const uint8_t *srow = map + (size_t)y * linesize;
			uint8_t *drow = &f->bgra[(size_t)y * w * 4];
			for (uint32_t x = 0; x < w; x++) {
				drow[x*4+0] = f->lut[srow[x*4+0]];
				drow[x*4+1] = f->lut[srow[x*4+1]];
				drow[x*4+2] = f->lut[srow[x*4+2]];
				drow[x*4+3] = srow[x*4+3];
			}
		}
		gs_stagesurface_unmap(f->stage);
	}

	// 3) entrega frame al worker (no bloquea)
	{
		std::lock_guard<std::mutex> lk(f->in_mtx);
		f->in_bgra = f->bgra; f->in_w = w; f->in_h = h; f->in_new = true;
	}
	f->in_cv.notify_one();

	// 4) toma juntos el último alpha y el frame que produjo ese alpha. Usar la
	// máscara anterior sobre el frame actual causaba el rastro al moverse.
	std::vector<float> alpha;
	std::vector<uint8_t> render_bgra;
	float fg_mean[3] = {0, 0, 0};
	bool fg_valid = false;
	{
		std::lock_guard<std::mutex> lk(f->out_mtx);
		if (f->out_w == (int)w && f->out_h == (int)h && !f->out_alpha.empty() &&
		    f->out_bgra.size() == (size_t)w * h * 4) {
			alpha = f->out_alpha;
			render_bgra = f->out_bgra;
		}
		for (int c = 0; c < 3; c++) fg_mean[c] = f->out_fg_mean[c];
		fg_valid = f->out_fg_valid;
	}
	if (alpha.empty()) { obs_source_skip_video_filter(f->context); return; } // aún sin máscara

	// 4b) ganancias del auto-match (EMA + LUTs por canal; identidad si off)
	update_auto_gains(f, fg_mean, fg_valid);

	// 5) compón
	std::vector<uint8_t> outbuf((size_t)w * h * 4);
	int mode = f->mode.load();
	if (mode == 1) {
		std::vector<uint8_t> blurred((size_t)w * h * 4);
		box_blur_bgra(render_bgra.data(), blurred.data(), w, h, std::max(1, (int)f->blur.load()));
		for (size_t i = 0; i < (size_t)w * h; i++) {
			float a = alpha[i];
			for (int c = 0; c < 3; c++) {
				float fg = render_bgra[i*4+c], bg = blurred[i*4+c];
				outbuf[i*4+c] = (uint8_t)std::clamp(bg + (fg-bg)*a, 0.f, 255.f);
			}
			outbuf[i*4+3] = 255;
		}
	} else {
		// auto_lut = ganancias del match por canal (identidad si esta off,
		// misma rama y coste). render_bgra es post-LUT-manual, pre-auto ->
		// las stats del fg no ven este ajuste (sin bucle de realimentacion).
		const uint8_t *lb = f->auto_lut[0], *lg = f->auto_lut[1], *lr = f->auto_lut[2];
		for (size_t i = 0; i < (size_t)w * h; i++) {
			float a = alpha[i];
			outbuf[i*4+0] = (uint8_t)std::clamp(lb[render_bgra[i*4+0]]*a, 0.f, 255.f);
			outbuf[i*4+1] = (uint8_t)std::clamp(lg[render_bgra[i*4+1]]*a, 0.f, 255.f);
			outbuf[i*4+2] = (uint8_t)std::clamp(lr[render_bgra[i*4+2]]*a, 0.f, 255.f);
			outbuf[i*4+3] = (uint8_t)std::clamp(a*255.f, 0.f, 255.f);
		}
	}

	// 6) sube y dibuja
	gs_texture_set_image(f->out_tex, outbuf.data(), w * 4, false);
	gs_effect_t *def = obs_get_base_effect(mode == 1 ? OBS_EFFECT_DEFAULT
							 : OBS_EFFECT_PREMULTIPLIED_ALPHA);
	gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
	gs_effect_set_texture(image, f->out_tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(f->out_tex, 0, w, h);
}

static void box_blur_bgra(const uint8_t *in, uint8_t *out, int W, int H, int radius)
{
	int r = std::max(1, radius / 3);
	std::vector<float> tmp((size_t)W * H * 3);
	for (int y = 0; y < H; y++)
		for (int c = 0; c < 3; c++) {
			float acc = 0; int cnt = 0;
			for (int x = -r; x <= r; x++) { acc += in[((size_t)y*W+std::clamp(x,0,W-1))*4+c]; cnt++; }
			for (int x = 0; x < W; x++) {
				tmp[((size_t)y*W+x)*3+c] = acc / cnt;
				acc += in[((size_t)y*W+std::clamp(x+r+1,0,W-1))*4+c]
				     - in[((size_t)y*W+std::clamp(x-r,0,W-1))*4+c];
			}
		}
	for (int x = 0; x < W; x++)
		for (int c = 0; c < 3; c++) {
			float acc = 0; int cnt = 0;
			for (int y = -r; y <= r; y++) { acc += tmp[((size_t)std::clamp(y,0,H-1)*W+x)*3+c]; cnt++; }
			for (int y = 0; y < H; y++) {
				out[((size_t)y*W+x)*4+c] = (uint8_t)std::clamp(acc / cnt, 0.f, 255.f);
				acc += tmp[((size_t)std::clamp(y+r+1,0,H-1)*W+x)*3+c]
				     - tmp[((size_t)std::clamp(y-r,0,H-1)*W+x)*3+c];
			}
		}
}

struct bg_enum_ctx {
	obs_property_t *list;
	obs_source_t *parent, *target;
};

static bool add_bg_source(void *param, obs_source_t *src)
{
	auto *ctx = static_cast<bg_enum_ctx *>(param);
	if (src == ctx->parent || src == ctx->target) return true;
	if (!(obs_source_get_output_flags(src) & OBS_SOURCE_VIDEO)) return true;
	const char *name = obs_source_get_name(src);
	if (name && *name) obs_property_list_add_string(ctx->list, name, name);
	return true;
}

static bool am_mode_changed(obs_properties_t *props, obs_property_t *, obs_data_t *s)
{
	// el auto-match solo tiene sentido en modo transparente (en blur el fondo
	// es la propia camara desenfocada: ya es coherente por definicion)
	bool transp = obs_data_get_int(s, "mode") == 0;
	obs_property_set_visible(obs_properties_get(props, "auto_match"), transp);
	obs_property_set_visible(obs_properties_get(props, "bg_source"), transp);
	obs_property_set_visible(obs_properties_get(props, "match_strength"), transp);
	return true;
}

static obs_properties_t *am_props(void *data)
{
	auto *f = static_cast<am_filter *>(data);
	obs_properties_t *p = obs_properties_create();
	obs_property_t *m = obs_properties_add_list(p, "mode", obs_module_text("Mode"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(m, obs_module_text("Mode.Transparent"), 0);
	obs_property_list_add_int(m, obs_module_text("Mode.Blur"), 1);
	obs_property_set_modified_callback(m, am_mode_changed);
	obs_properties_add_float_slider(p, "blur", obs_module_text("BlurStrength"), 4, 60, 1);
	// igualar luz con el fondo (solo modo transparente)
	obs_properties_add_bool(p, "auto_match", obs_module_text("AutoMatch"));
	obs_property_t *bl = obs_properties_add_list(p, "bg_source", obs_module_text("BgSource"),
			OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(bl, obs_module_text("BgSource.None"), "");
	bg_enum_ctx ctx = {bl, f ? obs_filter_get_parent(f->context) : nullptr,
			       f ? obs_filter_get_target(f->context) : nullptr};
	obs_enum_scenes(add_bg_source, &ctx);   // las escenas no salen por enum_sources
	obs_enum_sources(add_bg_source, &ctx);
	obs_properties_add_float_slider(p, "match_strength", obs_module_text("MatchStrength"), 0.0, 1.0, 0.05);
	obs_properties_add_float_slider(p, "gain", obs_module_text("Gain"), 0.5, 4.0, 0.05);
	obs_properties_add_float_slider(p, "gamma", obs_module_text("Gamma"), 0.4, 1.6, 0.05);
	obs_properties_add_float_slider(p, "alpha_gamma", obs_module_text("AlphaGamma"), 0.4, 1.5, 0.05);
	obs_property_t *q = obs_properties_add_list(p, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(q, obs_module_text("Quality.Fast"), 384);
	obs_property_list_add_int(q, obs_module_text("Quality.Medium"), 512);
	obs_property_list_add_int(q, obs_module_text("Quality.High"), 720);
	obs_property_t *d = obs_properties_add_list(p, "detail_ratio", obs_module_text("Detail"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(d, obs_module_text("Detail.Fast"), 40);
	obs_property_list_add_int(d, obs_module_text("Detail.Balanced"), 75);
	obs_property_list_add_int(d, obs_module_text("Detail.Max"), 100);
	obs_properties_add_path(p, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE,
				obs_module_text("ModelFilter"), nullptr);
	return p;
}

static void am_defaults(obs_data_t *s)
{
	obs_data_set_default_int(s, "mode", 0);
	obs_data_set_default_double(s, "blur", 22);
	obs_data_set_default_double(s, "gain", 1.0);
	obs_data_set_default_double(s, "gamma", 1.0);
	obs_data_set_default_double(s, "alpha_gamma", 0.75);
	obs_data_set_default_int(s, "quality", 512);
	obs_data_set_default_int(s, "detail_ratio", 75);
	obs_data_set_default_bool(s, "auto_match", false);
	obs_data_set_default_string(s, "bg_source", "");
	obs_data_set_default_double(s, "match_strength", 0.7);
}

static struct obs_source_info am_info = {};

bool obs_module_load(void)
{
	am_info.id = "obs_ai_matting";
	am_info.type = OBS_SOURCE_TYPE_FILTER;
	am_info.output_flags = OBS_SOURCE_VIDEO;
	am_info.get_name = am_get_name;
	am_info.create = am_create;
	am_info.destroy = am_destroy;
	am_info.video_render = am_render;
	am_info.update = am_update;
	am_info.get_properties = am_props;
	am_info.get_defaults = am_defaults;
	obs_register_source(&am_info);
	blog(LOG_INFO, "[obs-ai-matting] cargado");
	return true;
}
