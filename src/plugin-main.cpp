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
};

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

// corre RVM sobre buf BGRA(W,H) -> out (W*H alpha). Solo el worker llama esto.
static void run_matting(am_filter *f, const uint8_t *buf, int W, int H, std::vector<float> &out)
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
		run_matting(f, local.data(), W, H, alpha);
		if (!alpha.empty()) {
			std::lock_guard<std::mutex> lk(f->out_mtx);
			f->out_alpha = std::move(alpha);
			f->out_bgra = std::move(local);
			f->out_w = W; f->out_h = H;
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

// ---------- OBS ----------
static const char *am_get_name(void *) { return "AI Background (Matting)"; }

static void am_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<am_filter *>(data);
	f->mode = (int)obs_data_get_int(s, "mode");
	f->blur = obs_data_get_double(s, "blur");
	f->alpha_gamma = obs_data_get_double(s, "alpha_gamma");
	f->matte_long = (int)obs_data_get_int(s, "quality");
	f->detail_ratio = (int)obs_data_get_int(s, "detail_ratio");
	rebuild_lut(f, obs_data_get_double(s, "gain"), obs_data_get_double(s, "gamma"));

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
	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
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
	obs_enter_graphics();
	if (f->texrender) gs_texrender_destroy(f->texrender);
	if (f->stage) gs_stagesurface_destroy(f->stage);
	if (f->out_tex) gs_texture_destroy(f->out_tex);
	obs_leave_graphics();
	delete f->session;
	delete f->env;
	delete f;
}

static void am_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<am_filter *>(data);
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
	{
		std::lock_guard<std::mutex> lk(f->out_mtx);
		if (f->out_w == (int)w && f->out_h == (int)h && !f->out_alpha.empty() &&
		    f->out_bgra.size() == (size_t)w * h * 4) {
			alpha = f->out_alpha;
			render_bgra = f->out_bgra;
		}
	}
	if (alpha.empty()) { obs_source_skip_video_filter(f->context); return; } // aún sin máscara

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
		for (size_t i = 0; i < (size_t)w * h; i++) {
			float a = alpha[i];
			outbuf[i*4+0] = (uint8_t)std::clamp(render_bgra[i*4+0]*a, 0.f, 255.f);
			outbuf[i*4+1] = (uint8_t)std::clamp(render_bgra[i*4+1]*a, 0.f, 255.f);
			outbuf[i*4+2] = (uint8_t)std::clamp(render_bgra[i*4+2]*a, 0.f, 255.f);
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

static obs_properties_t *am_props(void *)
{
	obs_properties_t *p = obs_properties_create();
	obs_property_t *m = obs_properties_add_list(p, "mode", "Fondo", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(m, "Transparente (pon fondo detras en OBS)", 0);
	obs_property_list_add_int(m, "Desenfoque (blur)", 1);
	obs_properties_add_float_slider(p, "blur", "Intensidad de blur", 4, 60, 1);
	obs_properties_add_float_slider(p, "gain", "Brillo (gain)", 0.5, 4.0, 0.05);
	obs_properties_add_float_slider(p, "gamma", "Gamma (<1 aclara)", 0.4, 1.6, 0.05);
	obs_properties_add_float_slider(p, "alpha_gamma", "Dureza de recorte", 0.4, 1.5, 0.05);
	obs_property_t *q = obs_properties_add_list(p, "quality", "Calidad de recorte", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(q, "Rapida (384)", 384);
	obs_property_list_add_int(q, "Media (512)", 512);
	obs_property_list_add_int(q, "Alta (720)", 720);
	obs_property_t *d = obs_properties_add_list(p, "detail_ratio", "Detalle interno (manos y cabello)", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(d, "Rapido (0.40)", 40);
	obs_property_list_add_int(d, "Equilibrado (0.75)", 75);
	obs_property_list_add_int(d, "Maximo (1.00)", 100);
	obs_properties_add_path(p, "model_path", "Modelo RVM (.onnx)", OBS_PATH_FILE,
				"ONNX (*.onnx);;Todos (*.*)", nullptr);
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
