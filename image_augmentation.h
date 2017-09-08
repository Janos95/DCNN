// Fast data augmentation routines for images
// Author: Max Schwarz <max.schwarz@uni-bonn.de>

#ifndef IMAGE_AUGMENTATION_H
#define IMAGE_AUGMENTATION_H

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>

#include <x86intrin.h>

namespace image_augmentation
{

/**
 * @brief Image
 *
 * Image in (CxHxW) format.
 **/
template<class T>
struct Image
{
	enum class Format
	{
		Invalid,
		CHW,
		HWC
	};

	enum class RGBMode
	{
		Invalid,
		RGB,
		BGR
	};

	Format format;
	RGBMode rgbMode;
	std::size_t stride[2];  //!< stride of the two *outer* dimensions (CH or HW)
	std::size_t height;
	std::size_t width;
	std::size_t channels;
	T* data;
};

struct TransformerSettings
{
	bool mirrorH = true;    //!< Mirror image horizontally
	bool mirrorV = false;   //!< Mirror image vertically

	struct {
		float H = 0.02f;    //!< Standard deviation for H offset (normalized)
		float S = 0.04f;    //!< Standard deviation for S offset (normalized)
		float V = 0.08f;    //!< Standard deviation for V offset (normalized)
	} HSV;

	float scale = 0.02f;    //!< Standard deviation for scale (normalized)
};

class Transformer
{
public:
	enum class Interpolation
	{
		Bilinear,
		Nearest
	};

	template<class RandomSource>
	Transformer(const TransformerSettings& settings, RandomSource& source);

	template<class T>
	void transformImage(Image<T>* img, bool doHSV = true, Interpolation interpolation = Interpolation::Bilinear);

	template<class T>
	void transformPoint(int width, int height, T x, T y, T* tx, T* ty);

	template<class T>
	void transformRectXcYcWH(int width, int height, T x, T y, T w, T h, T* tx, T* ty, T* tw, T* th);
private:
	bool m_mirrorH;
	bool m_mirrorV;

	float m_offsetH;
	float m_offsetS;
	float m_offsetV;

	float m_scale;
	float m_offsetX;
	float m_offsetY;
};

////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION

namespace impl
{

	template<class T>
	void RGB2HSV(T r, T g, T b, T &h, T &s, T &v)
	{
		T K = 0.f;

		if (g < b)
		{
			std::swap(g, b);
			K = static_cast<T>(-1.0);
		}

		if (r < g)
		{
			std::swap(r, g);
			K = static_cast<T>(-2.0 / 6.0) - K;
		}

		T chroma = r - std::min(g, b);
		h = fabs(K + (g - b) / (static_cast<T>(6.0) * chroma + static_cast<T>(1e-20)));
		s = chroma / (r + static_cast<T>(1e-20));
		v = r;
	}

	template<class T>
	T clamp(T arg)
	{
		return std::max<T>(0.0, std::min<T>(1.0, arg));
	}

	template<class T>
	void HSV2RGB(const T H, const T S, const T V, T& r, T& g, T& b)
	{
		__m128 w = _mm_set_ps(
			std::abs(H * 6 - 3) - 1, // r
			2 - std::abs(H * 6 - 2), // g
			2 - std::abs(H * 6 - 4), // b
			0.0f
		);

		__m128 lower = _mm_setzero_ps();
		__m128 ones = _mm_set_ps1(1.0f);

		// clamp r,g,b to [0,1]
		__m128 mask = _mm_cmpge_ps(w, lower);
		w = _mm_and_ps(w, mask);

		mask = _mm_cmpgt_ps(w, ones);
		w = _mm_or_ps(_mm_and_ps(ones, mask), _mm_andnot_ps(mask, w));

		// finally use S and V
		__m128 vec_s = _mm_set_ps1(S);
		__m128 vec_v = _mm_set_ps1(V);

		w = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(w, ones), vec_s), ones), vec_v);

		_MM_EXTRACT_FLOAT(r, w, 3);
		_MM_EXTRACT_FLOAT(g, w, 2);
		_MM_EXTRACT_FLOAT(b, w, 1);
	}
}

template<class T>
void Transformer::transformPoint(int width, int height, T x, T y, T* tx, T* ty)
{
	if(m_mirrorH)
		x = width - 1 - x;

	if(m_mirrorV)
		y = height - 1 - y;

	if(m_scale != 0)
	{
		x = m_scale * (float(x) - m_offsetX * width);
		y = m_scale * (float(y) - m_offsetY * height);
	}

	*tx = x;
	*ty = y;
}

template<class T>
void Transformer::transformRectXcYcWH(int width, int height,
	T x, T y, T w, T h,
	T* tx, T* ty, T* tw, T* th)
{
	transformPoint(width, height, x, y, tx, ty);
	*tw = m_scale * w;
	*th = m_scale * h;
}

template<class RandomSource>
Transformer::Transformer(const TransformerSettings& settings, RandomSource& source)
{
	std::bernoulli_distribution bernoulli;

	m_mirrorH = settings.mirrorH ? bernoulli(source) : false;
	m_mirrorV = settings.mirrorV ? bernoulli(source) : false;

	m_offsetH = std::normal_distribution<float>(0.0, settings.HSV.H)(source);
	m_offsetS = std::normal_distribution<float>(0.0, settings.HSV.S)(source);
	m_offsetV = std::normal_distribution<float>(0.0, settings.HSV.V)(source);

	m_scale = std::normal_distribution<float>(0.0, settings.scale)(source);

	// We only zoom *in*
	m_scale = std::abs(m_scale) + 1.0;

	// Scaling by >1.0 allows us to execute random crops.
	float availableBorder = 1.0 - (1.0 / m_scale);
	m_offsetX = std::uniform_real_distribution<float>(0.0, availableBorder)(source);
	m_offsetY = std::uniform_real_distribution<float>(0.0, availableBorder)(source);
}

template<class T>
void Transformer::transformImage(Image<T>* img, bool doHSV, Interpolation interpolation)
{
	std::size_t width = img->width;
	std::size_t height = img->height;
	std::size_t channels = img->channels;

	if(m_mirrorH)
	{
		// Mirror the image along the horizontal dimension
		if(img->format == Image<T>::Format::CHW)
		{
			for(std::size_t channel = 0; channel < channels; ++channel)
			{
				for(std::size_t y = 0; y < height; ++y)
				{
					T* rowPtr = img->data + channel * img->stride[0] + y * img->stride[1];

					for(std::size_t x = 0; x < width/2; ++x)
						std::swap(rowPtr[x], rowPtr[width - 1 - x]);
				}
			}
		}
		else if(img->format == Image<T>::Format::HWC)
		{
			for(std::size_t y = 0; y < height; ++y)
			{
				for(std::size_t x = 0; x < width/2; ++x)
				{
					T* srcPtr = img->data + y * img->stride[0] + x * img->stride[1];
					T* dstPtr = img->data + y * img->stride[0] + (width - 1 - x) * img->stride[1];

					for(std::size_t channel = 0; channel < channels; ++channel)
						std::swap(srcPtr[channel], dstPtr[channel]);
				}
			}
		}
	}

	if(m_mirrorV)
	{
		// Mirror the image along the vertical dimension
		if(img->format == Image<T>::Format::CHW)
		{
			std::vector<T> tmp(img->stride[1]);

			for(std::size_t channel = 0; channel < channels; ++channel)
			{
				for(std::size_t y = 0; y < height/2; ++y)
				{
					memcpy(
						tmp.data(),
						img->data + channel * img->stride[0] + y * img->stride[1],
						img->stride[1] * sizeof(T)
					);

					memcpy(
						img->data + channel * img->stride[0] + y * img->stride[1],
						img->data + channel * img->stride[0] + (height - 1 - y) * img->stride[1],
						img->stride[1] * sizeof(T)
					);

					memcpy(
						img->data + channel * img->stride[0] + (height - 1 - y) * img->stride[1],
						tmp.data(),
						img->stride[1] * sizeof(T)
					);
				}
			}
		}
		else if(img->format == Image<T>::Format::HWC)
		{
			for(std::size_t y = 0; y < height/2; ++y)
			{
				T* srcPtr = img->data + y * img->stride[0];
				T* dstPtr = img->data + (height - 1 - y) * img->stride[0];

				for(std::size_t x = 0; x < width; ++x)
				{
					for(std::size_t channel = 0; channel < channels; ++channel)
						std::swap(srcPtr[x * img->stride[1] + channel], dstPtr[x * img->stride[1] + channel]);
				}
			}
		}
	}

	// HSV augmentation
	if(doHSV && (m_offsetH != 0.0f || m_offsetS != 0.0f || m_offsetV != 0.0f))
	{
		if(img->channels != 3)
			throw std::logic_error("Need 3 channels for HSV augmentation");

		if(img->format == Image<T>::Format::CHW)
		{
			T* bPane;
			T* gPane;
			T* rPane;

			switch(img->rgbMode)
			{
				case Image<T>::RGBMode::BGR:
					bPane = img->data;
					gPane = img->data + 1 * img->stride[0];
					rPane = img->data + 2 * img->stride[0];
					break;
				case Image<T>::RGBMode::RGB:
					rPane = img->data;
					gPane = img->data + 1 * img->stride[0];
					bPane = img->data + 2 * img->stride[0];
					break;
				default:
					throw std::logic_error("Please specify the rgbMode field");
			}

			for(std::size_t y = 0; y < height; ++y)
			{
				for(std::size_t x = 0; x < width; ++x)
				{
					std::size_t off = y * img->stride[1] + x;
					T r = rPane[off];
					T g = gPane[off];
					T b = bPane[off];

					T H, S, V;
					impl::RGB2HSV(r, g, b, H, S, V);

					if(!std::isfinite(H) || !std::isfinite(S) || !std::isfinite(V))
					{
						fprintf(stderr, "HSV has NAN!\n");
						std::abort();
					}

					// H wraps around
					H = H + m_offsetH;
					if(H < 0)
						H += static_cast<T>(1.0);
					else if(H > 1)
						H -= static_cast<T>(1.0);

					// S,V are clamped to [0,1]
					S = impl::clamp(S + m_offsetS);
					V = impl::clamp(V + m_offsetV);

					// Convert back to RGB
					impl::HSV2RGB(H, S, V, r, g, b);

					if(!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
					{
						fprintf(stderr, "RGB has NAN!\n");
						std::abort();
					}

					// and write into dest.
					rPane[off] = r;
					gPane[off] = g;
					bPane[off] = b;
				}
			}
		}
		else if(img->format == Image<T>::Format::HWC)
		{
			for(std::size_t y = 0; y < height; ++y)
			{
				for(std::size_t x = 0; x < width; ++x)
				{
					std::size_t off = y * img->stride[0] + x * img->stride[1];

					T r, g, b;
					switch(img->rgbMode)
					{
						case Image<T>::RGBMode::BGR:
							r = img->data[off + 2];
							g = img->data[off + 1];
							b = img->data[off + 0];
							break;
						case Image<T>::RGBMode::RGB:
							r = img->data[off + 2];
							g = img->data[off + 1];
							b = img->data[off + 0];
							break;
						default:
							throw std::logic_error("Please specify rgbMode field");
					}

					T H, S, V;
					impl::RGB2HSV(r, g, b, H, S, V);

					if(!std::isfinite(H) || !std::isfinite(S) || !std::isfinite(V))
					{
						fprintf(stderr, "HSV has NAN!\n");
						std::abort();
					}

					// H wraps around
					H = H + m_offsetH;
					if(H < 0)
						H += static_cast<T>(1.0);
					else if(H > 1)
						H -= static_cast<T>(1.0);

					// S,V are clamped to [0,1]
					S = impl::clamp(S + m_offsetS);
					V = impl::clamp(V + m_offsetV);

					// Convert back to RGB
					impl::HSV2RGB(H, S, V, r, g, b);

					if(!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
					{
						fprintf(stderr, "RGB has NAN!\n");
						std::abort();
					}

					// and write into dest.
					img->data[off + 2] = r;
					img->data[off + 1] = g;
					img->data[off + 0] = b;
				}
			}
		}
	}

	// Crop & scale
	if(m_scale != 1.0f)
	{
		// Can't avoid complete copy :-( TODO: Could we elide this?
		std::vector<T> tmp(height*width*channels);

		if(img->format == Image<T>::Format::CHW)
		{
			// Copy source image
			for(std::size_t channel = 0; channel < channels; ++channel)
			{
				for(std::size_t y = 0; y < height; ++y)
				{
					memcpy(
						tmp.data() + channel*height*width + y*width,
						img->data + channel*img->stride[0] + y*img->stride[1],
						width * sizeof(T)
					);
				}
			}

			// Go through *target* image and calculate pixel values
			switch(interpolation)
			{
				case Interpolation::Bilinear:
					for(std::size_t channel = 0; channel < channels; ++channel)
					{
						for(std::size_t y = 0; y < height; ++y)
						{
							float src_y = m_offsetY * height + y / m_scale;

							for(std::size_t x = 0; x < width; ++x)
							{
								float src_x = m_offsetX * width + x / m_scale;

								int ix = (int)src_x;
								int iy = (int)src_y;

								T a = src_x - (T)ix;
								T c = src_y - (T)iy;

								ix = std::max(0, std::min<int>(width-2, ix));
								iy = std::max(0, std::min<int>(height-2, iy));

								// bilinear interpolation
								T value =
									(tmp[channel*height*width + iy*width + ix] * (1 - a)
									+ tmp[channel*height*width + iy*width + ix + 1] * a) * (1 - c)
									+ (tmp[channel*height*width + (iy+1)*width + ix] * (1 - a)
									+ tmp[channel*height*width + (iy+1)*width + ix + 1] * a) * c;

								img->data[channel*img->stride[0] + y*img->stride[1] + x] = value;
							}
						}
					}
					break;
				case Interpolation::Nearest:
				{
					for(std::size_t channel = 0; channel < channels; ++channel)
					{
						for(std::size_t y = 0; y < height; ++y)
						{
							float src_y = m_offsetY * height + y / m_scale;

							for(std::size_t x = 0; x < width; ++x)
							{
								float src_x = m_offsetX * width + x / m_scale;

								int ix = std::round(src_x);
								int iy = std::round(src_y);

								ix = std::max(0, std::min<int>(width-1, ix));
								iy = std::max(0, std::min<int>(height-1, iy));

								// bilinear interpolation
								T value = tmp[channel*height*width + iy*width + ix];

								img->data[channel*img->stride[0] + y*img->stride[1] + x] = value;
							}
						}
					}
					break;
				}
				default:
					throw std::logic_error("Invalid interpolation mode");
			}
		}
		else if(img->format == Image<T>::Format::HWC)
		{
			// Copy source image
			for(std::size_t y = 0; y < height; ++y)
			{
				for(std::size_t x = 0; x < width; ++x)
				{
					for(std::size_t channel = 0; channel < 3; ++channel)
					{
						tmp[y*width*3 + x*3 + channel] =
							img->data[y*img->stride[0] + x*img->stride[1] + channel];
					}
				}
			}

			// Go through *target* image and calculate pixel values
			switch(interpolation)
			{
				case Interpolation::Bilinear:
					for(std::size_t y = 0; y < height; ++y)
					{
						float src_y = m_offsetY * height + float(y) / m_scale;

						for(std::size_t x = 0; x < width; ++x)
						{
							float src_x = m_offsetX * width + float(x) / m_scale;

							int ix = (int)src_x;
							int iy = (int)src_y;

							T a = src_x - (T)ix;
							T c = src_y - (T)iy;

							ix = std::max(0, std::min<int>(width-2, ix));
							iy = std::max(0, std::min<int>(height-2, iy));

							for(std::size_t channel = 0; channel < channels; ++channel)
							{
								// bilinear interpolation
								T value =
									(tmp[iy*width*3     + ix*3     + channel] * (1.0f - a)
										+ tmp[iy*width*3     + (ix+1)*3 + channel] * a) * (1.0f - c)
									+ (tmp[(iy+1)*width*3 + ix*3     + channel] * (1.0f - a)
										+ tmp[(iy+1)*width*3 + (ix+1)*3 + channel] * a) * c;

								img->data[y*img->stride[0] + x*img->stride[1] + channel] = value;
							}
						}
					}
					break;
				case Interpolation::Nearest:
					for(std::size_t y = 0; y < height; ++y)
					{
						float src_y = m_offsetY * height + float(y) / m_scale;

						for(std::size_t x = 0; x < width; ++x)
						{
							float src_x = m_offsetX * width + float(x) / m_scale;

							int ix = std::round(src_x);
							int iy = std::round(src_y);

							ix = std::max(0, std::min<int>(width-1, ix));
							iy = std::max(0, std::min<int>(height-1, iy));

							for(std::size_t channel = 0; channel < channels; ++channel)
							{
								T value = tmp[iy*width*channels     + ix*channels     + channel];
								img->data[y*img->stride[0] + x*img->stride[1] + channel] = value;
							}
						}
					}
					break;
				default:
					throw std::logic_error("Invalid interpolation mode");
			}
		}
	}
}

}

#endif
