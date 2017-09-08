
#include <TH.h>
#include <luaT.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>


#include "SlowSpatialConvolution.h"


static int im2col(lua_State *L)
{
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    std::size_t kW = luaL_checkinteger(L, 2); 
    std::size_t kH = luaL_checkinteger(L, 3);
    std::size_t C = THDoubleTensor_size(image, 0); //number of input channels
    std::size_t H = THDoubleTensor_size(image, 1); //input image height
    std::size_t W = THDoubleTensor_size(image, 2); // input image width
    
    std::size_t wOutputImage = W-kW+1;
    std::size_t hOutputImage = H-kH+1;
    THDoubleTensor* outputImage = THDoubleTensor_newWithSize2d(kW*kH*C, wOutputImage*hOutputImage);
    std::size_t s0 = THDoubleTensor_stride(outputImage, 0);
    std::size_t s1 = THDoubleTensor_stride(outputImage, 1);
    std::size_t t0 = THDoubleTensor_stride(image, 0);
    std::size_t t1 = THDoubleTensor_stride(image, 1);
    std::size_t t2 = THDoubleTensor_stride(image, 2);
    std::size_t c,k,l,i,j;
    for(c = 0; c <= C-1; c++){
        for (k = 0; k <= kW - 1; k++){
            for (l = 1; l <= kH; l++){
                for (i = 0; i <= hOutputImage - 1; i++){
                    for (j = 1; j <= wOutputImage; j++){
                      outputImage[(c*kW*kH+k*kH+l-1)*s0+s1*(i*wOutputImage+j-1)] = image[(c)*t0+(i+l-1)*t1+(j+k-1)*t2];
                    }
                }
            }
        }
    }
    luaT_pushudata(L, outputImage, "torch.DoubleTensor");

    return 1;
}


static const struct luaL_Reg Global_funcs[] = {
	{"im2col", im2col},
	{NULL, NULL}
};

extern "C"
{

DLL_EXPORT int luaopen_libslowspatialconvolution(lua_State* L)
{
	lua_newtable(L);
	luaT_setfuncs(L, Global_funcs, 0);
	lua_setglobal(L, "slowspatialconvolution");

	return 0;
}

}
