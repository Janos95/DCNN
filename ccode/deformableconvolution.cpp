
#include <TH.h>
#include <luaT.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <math.h>
#include <assert.h> 
#include <random>
#include <stdexcept>
#include <iostream>
#include <algorithm>


#include "deformableconvolution.h"


/* @brief Helper function for bilinearInterp:
 * projects a coordinate x to 0 if x < 0 and to range if x > range
 */
inline double projection(double x, double range){
    if(x < 0)
        return 0;
    if(x > range)
        return range;
    return x;
}

static double isPositive(double a){
    if(a >= 0) return 1.0;
    return 0.0;
}

/* @brief interpolation using the formula (3) + (4) from [1].
 * We project the point (px,py) onto the image, interpolate the projected point
 * (px_projected,py_projected) and then multiply by 
 * (1-abs(px-px_projected))(1-abs(py-py_projected)) to correct.
 * @param input 3D-Tensor representation of an image
 * @param c the image channel we want to use
 * @params py,px the (fractional) (y,x)-coordinates where we want to interpolate
 * @return retval the interpolated value of image(py,px)
 */
static double bilinearInterp(THDoubleTensor* input, THLongTensor* bi, THDoubleTensor* bw, long c, double py, double px, std::size_t i, std::size_t j, long save_buffer){
    std::size_t H = THDoubleTensor_size(input, 1); // input image height
    std::size_t W = THDoubleTensor_size(input, 2); // input image width
    
    double *data = THDoubleTensor_data(input);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    
    
    double retval = 0;
    
    // TODO: If this is necessary, fill bw and bi appropriately
//     if((px <= -1) or (px >= W) or (py <= -1) or (py >= H)){
//         return retval;
//     }
    
    double px_projected = projection(px, double(W-1));
    double py_projected = projection(py, double(H-1));
    
    long ax = long(floor(px_projected)); 
    long ay = long(floor(py_projected));
    
    if(ax >= W-1)
        ax = W-2;
    if(ay >= H-1)
        ay = H-2;
    
    //dim(bw) = H_{out}*W_{out} x c1*kH*kW x 4
    double w[4] = {0,0,0,0};
    
    w[0] = (1-std::abs(py_projected - ay));
    w[1] = (1-std::abs(px_projected - ax));
    w[2] = (1-std::abs(py_projected - (ay+1)));
    w[3] = (1-std::abs(px_projected - (ax+1)));

    
    
//     if(px_projected != px or py_projected != py){
//         w[0] *= std::max(0.0, 1-std::abs(py-py_projected));
//         w[1] *= std::max(0.0, 1-std::abs(px-px_projected));
//         w[2] *= std::max(0.0, 1-std::abs(py-py_projected));
//         w[3] *= std::max(0.0, 1-std::abs(px-px_projected));
//     }

    
    retval = w[0]*w[1]*data[s0*c+s1*ay+s2*ax]
            +w[0]*w[3]*data[s0*c+s1*ay+s2*(ax+1)]
            +w[2]*w[1]*data[s0*c+s1*(ay+1)+s2*ax]
            +w[2]*w[3]*data[s0*c+s1*(ay+1)+s2*(ax+1)];
    
    if(save_buffer){
        long *bi_data = THLongTensor_data(bi);
        double *bw_data = THDoubleTensor_data(bw);
        
        std::size_t t0 = THDoubleTensor_stride(bw, 0);
        std::size_t t1 = THDoubleTensor_stride(bw, 1);
        std::size_t t2 = THDoubleTensor_stride(bw, 2);
    
        std::size_t r0 = THLongTensor_stride(bi, 0);
        std::size_t r1 = THLongTensor_stride(bi, 1);
        std::size_t r2 = THLongTensor_stride(bi, 2);
        
        bi_data[r0*i + r1*j + r2*0] = c;
        bi_data[r0*i + r1*j + r2*1] = ay;
        bi_data[r0*i + r1*j + r2*2] = ax;
        
        bw_data[t0*i + t1*j + t2*0] = w[0];
        bw_data[t0*i + t1*j + t2*1] = w[1];
        bw_data[t0*i + t1*j + t2*2] = w[2];
        bw_data[t0*i + t1*j + t2*3] = w[3];
    }
    
    return retval;
}

/* @brief lua-wrapper for the function bilinearInterp
 * @param L lua_State which contains the parameters 3D-input-image, input channel, interpolation point parameters py, px
 * @return 1 hands the bilinear Interpolation retval of image(c,py,px) back to lua_State
 */
static int bilinearInterpolation(lua_State *L){
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 2, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 3, "torch.DoubleTensor");
    std::size_t c = luaL_checkinteger(L, 4)-1; 
    double py = luaL_checknumber(L, 5)-1;
    double px = luaL_checknumber(L, 6)-1;
    std::size_t i = luaL_checkinteger(L, 7)-1;
    std::size_t j = luaL_checkinteger(L, 8)-1;
    long save_buffer = luaL_checkinteger(L, 9);
    
    double retval = bilinearInterp(image, bi, bw, c, py, px, i, j, save_buffer);
    
    lua_pushnumber(L, retval);
    return 1;
}

/* @brief turns 3D-batches from the 3D-input-image into columns in the 2D-output-matrix,
 * taking into account offset data
 * @param L lua_State which contains the input image, the offset data, and batch size parameters kH and kW
 * @return 1 hands im2col(input) back to the lua_State
 */
static int im2col(lua_State *L)
{
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor"); //dimension(image) = C x H x W
    THDoubleTensor *offsets = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor"); //dimension(offsets) = H_out x W_out x kH x kW x 2
    std::size_t kH = luaL_checkinteger(L, 3); //kernel height
    std::size_t kW = luaL_checkinteger(L, 4); //kernel width
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 5, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 6, "torch.DoubleTensor");
    long save_buffer = luaL_checkinteger(L, 7);
    
    std::size_t C = THDoubleTensor_size(image, 0); //number of input channels
    std::size_t H = THDoubleTensor_size(image, 1); //input image height
    std::size_t W = THDoubleTensor_size(image, 2); //input image width
    
    std::size_t a = THDoubleTensor_size(offsets, 0); //hOutputImage
    std::size_t b = THDoubleTensor_size(offsets, 1); //wOutputImage
    std::size_t ce = THDoubleTensor_size(offsets, 2); //kH
    std::size_t d = THDoubleTensor_size(offsets, 3); //kW
    std::size_t e = THDoubleTensor_size(offsets, 4); //2    
    
    std::size_t wOutputImage = W - kW + 1;
    std::size_t hOutputImage = H - kH + 1;
    
//     std::cout << C << " " << H << " " << W << std::endl;
    THDoubleTensor* outputImage = THDoubleTensor_newWithSize2d(kW*kH*C, wOutputImage*hOutputImage); //im2col(input)
    
    std::size_t s0 = THDoubleTensor_stride(outputImage, 0);
    std::size_t s1 = THDoubleTensor_stride(outputImage, 1);
    
    std::size_t t0 = THDoubleTensor_stride(offsets, 0);
    std::size_t t1 = THDoubleTensor_stride(offsets, 1);
    std::size_t t2 = THDoubleTensor_stride(offsets, 2);
    std::size_t t3 = THDoubleTensor_stride(offsets, 3);
    std::size_t t4 = THDoubleTensor_stride(offsets, 4);
    
    std::size_t r0 = THDoubleTensor_stride(image, 0);
    std::size_t r1 = THDoubleTensor_stride(image, 1);
    std::size_t r2 = THDoubleTensor_stride(image, 2);
    
    long c;
    std::size_t k,l,i,j;
    
    double* odata = THDoubleTensor_data(outputImage);
    double* idata = THDoubleTensor_data(image);
    double* offsets_data = THDoubleTensor_data(offsets);
    double x_offset = 0;
    double y_offset = 0;
    
    for(c = 0; c <= C - 1; c++){
        for (i = 0; i <= hOutputImage - 1; i++){
            for (j = 0; j <= wOutputImage - 1; j++){
                for (l = 0; l <= kH - 1; l++){
                    for (k = 0; k <= kW - 1 ; k++){
                        assert(i < d);
                        assert(j < e);
                        assert(l < b);
                        assert(k < ce);
                        assert(1 < a);
                        y_offset = offsets_data[0*t0+l*t1+k*t2+i*t3+j*t4];
                        x_offset = offsets_data[1*t0+l*t1+k*t2+i*t3+j*t4];
                        if(y_offset >= 5000 or x_offset >= 5000)
                            std::cout << y_offset << " " << x_offset << std::endl;
                        assert(y_offset < 5000 and x_offset < 5000);
                        odata[(c*kW*kH+l*kW+k)*s0+s1*(i*wOutputImage+j)] =
                            bilinearInterp(image, 
                                           bi, 
                                           bw, 
                                           c, 
                                           i+l+y_offset, 
                                           j+k+x_offset, 
                                           c*kW*kH+l*kW+k,
                                           i*wOutputImage+j, 
                                           save_buffer);
                    }
                }
            }
        }
    }
    luaT_pushudata(L, outputImage, "torch.DoubleTensor");

    return 1;
}


static int update_grad_input(lua_State *L){
    THDoubleTensor *gradIm2col = (THDoubleTensor*)luaT_checkudata(L, 1, 
"torch.DoubleTensor");
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 2, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 3, 
"torch.DoubleTensor");
    std::size_t C = luaL_checkinteger(L, 4);
    std::size_t H = luaL_checkinteger(L, 5);
    std::size_t W = luaL_checkinteger(L, 6);
    
    THDoubleTensor *gradInput = THDoubleTensor_newWithSize3d(C,H,W);
    
    double *gradInput_data = THDoubleTensor_data(gradInput);
    double *gradIm2col_data = THDoubleTensor_data(gradIm2col);
    long *bi_data = THLongTensor_data(bi);
    double *bw_data = THDoubleTensor_data(bw);
    
    std::size_t H_im2col = THDoubleTensor_size(gradIm2col, 0);
    std::size_t W_im2col = THDoubleTensor_size(gradIm2col, 1);
    
    std::size_t s0 = THDoubleTensor_stride(gradIm2col, 0);
    std::size_t s1 = THDoubleTensor_stride(gradIm2col, 1);

    std::size_t t0 = THDoubleTensor_stride(gradInput, 0);
    std::size_t t1 = THDoubleTensor_stride(gradInput, 1);
    std::size_t t2 = THDoubleTensor_stride(gradInput, 2);

    
    std::size_t r0 = THDoubleTensor_stride(bw, 0);
    std::size_t r1 = THDoubleTensor_stride(bw, 1);
    std::size_t r2 = THDoubleTensor_stride(bw, 2);
    
    std::size_t q0 = THLongTensor_stride(bi, 0);
    std::size_t q1 = THLongTensor_stride(bi, 1);
    std::size_t q2 = THLongTensor_stride(bi, 2);
    
    assert(THLongTensor_size(bi, 0) == H_im2col);
    assert(THLongTensor_size(bi, 1) == W_im2col);
    assert(THLongTensor_size(bi, 2) == 3);

    std::size_t i,j,k;
    
    for(i = 0; i < C; i++)
        for(j = 0; j < H; j++)
            for(k = 0; k < W; k++)
                gradInput_data[t0*i+t1*j+t2*k] = 0;
    
    for(i=0; i<H_im2col; i++){
        for(j=0; j<W_im2col; j++){
            long c = bi_data[q0*i+q1*j+q2*0];
            long ay = bi_data[q0*i+q1*j+q2*1];
            long ax = bi_data[q0*i+q1*j+q2*2];
            
            double w0, w1, w2, w3;
            w0 = bw_data[r0*i+r1*j+r2*0]*bw_data[r0*i+r1*j+r2*1];
            w1 = bw_data[r0*i+r1*j+r2*0]*bw_data[r0*i+r1*j+r2*3];
            w2 = bw_data[r0*i+r1*j+r2*2]*bw_data[r0*i+r1*j+r2*1];
            w3 = bw_data[r0*i+r1*j+r2*2]*bw_data[r0*i+r1*j+r2*3];
            
            assert(c <= C-1);
            assert(ay <= H-1);
            assert(ax <= W-1);
            
            gradInput_data[c*t0+ay*t1+ax*t2] += 
                            w0 * gradIm2col_data[s0*i+s1*j];
            gradInput_data[c*t0+ay*t1+(ax+1)*t2] += 
                            w1 * gradIm2col_data[s0*i+s1*j];
            gradInput_data[c*t0+(ay+1)*t1+ax*t2] += 
                            w2 * gradIm2col_data[s0*i+s1*j];
            gradInput_data[c*t0+(ay+1)*t1+(ax+1)*t2] += 
                            w3 * gradIm2col_data[s0*i+s1*j];
        }
    }
   
    luaT_pushudata(L, gradInput, "torch.DoubleTensor");
    return 1;
}

static int grad_offset(lua_State *L){
    
    THDoubleTensor *input = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    THDoubleTensor *offset = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor");
    THDoubleTensor *weight = (THDoubleTensor*)luaT_checkudata(L, 3, "torch.DoubleTensor");
    THDoubleTensor *gradOutput = (THDoubleTensor*)luaT_checkudata(L, 4, "torch.DoubleTensor");
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 5, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 6, "torch.DoubleTensor");
    
    std::size_t C2 = THDoubleTensor_size(weight,0);
    std::size_t kH = THDoubleTensor_size(weight,2);
    std::size_t kW = THDoubleTensor_size(weight,3);
   
    std::size_t C1 = THDoubleTensor_size(input,0);
    std::size_t H = THDoubleTensor_size(input,1);
    std::size_t W = THDoubleTensor_size(input,2);
    
    std::size_t H_out = H - kH + 1;
    std::size_t W_out = W - kW + 1;
    
    THDoubleTensor *gradOffset = THDoubleTensor_newWithSize3d(2*kH*kW, H_out, W_out);
    
    double* input_data = THDoubleTensor_data(input);
    double* offset_data = THDoubleTensor_data(offset);
    double* weight_data = THDoubleTensor_data(weight);
    double* gradOutput_data = THDoubleTensor_data(gradOutput);
    long* bi_data = THLongTensor_data(bi);
    double* bw_data = THDoubleTensor_data(bw);
    
    double* gradOffset_data = THDoubleTensor_data(gradOffset);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    
    std::size_t t0 = THDoubleTensor_stride(offset, 0);
    std::size_t t1 = THDoubleTensor_stride(offset, 1);
    std::size_t t2 = THDoubleTensor_stride(offset, 2);
    std::size_t t3 = THDoubleTensor_stride(offset, 3);
    std::size_t t4 = THDoubleTensor_stride(offset, 4);
    
    std::size_t u0 = THDoubleTensor_stride(weight, 0);
    std::size_t u1 = THDoubleTensor_stride(weight, 1);
    std::size_t u2 = THDoubleTensor_stride(weight, 2);
    std::size_t u3 = THDoubleTensor_stride(weight, 3);
    
    std::size_t z0 = THDoubleTensor_stride(gradOutput, 0);
    std::size_t z1 = THDoubleTensor_stride(gradOutput, 1);
    std::size_t z2 = THDoubleTensor_stride(gradOutput, 2);

    std::size_t r0 = THLongTensor_stride(bi, 0);
    std::size_t r1 = THLongTensor_stride(bi, 1);
    std::size_t r2 = THLongTensor_stride(bi, 2);
   
    std::size_t q0 = THDoubleTensor_stride(bw, 0);
    std::size_t q1 = THDoubleTensor_stride(bw, 1);
    std::size_t q2 = THDoubleTensor_stride(bw, 2);
    
    std::size_t v0 = THDoubleTensor_stride(gradOffset, 0);
    std::size_t v1 = THDoubleTensor_stride(gradOffset, 1);
    std::size_t v2 = THDoubleTensor_stride(gradOffset, 2);
    
    std::size_t c2, c1, i, j, k, l;
    
    for(i=0; i < 2*kH*kW; i++)
        for(j=0; j < H_out; j++)
            for(k=0; k < W_out; k++)
                gradOffset_data[i*v0 + j*v1 + k*v2] = 0.0;
    
    for(c2=0; c2 < C2; c2++){
        for(c1=0; c1 < C1; c1++){
            for(i=0; i < H_out; i++){
                for(j=0; j < W_out; j++){
                    for(k=0; k < kH; k++){
                        for(l=0; l < kW; l++){
                            long c = bi_data[r0*(c1*kH*kW+k*kW+l) + r1*(i*W_out+j) + 0*r2];
                            long ay = bi_data[r0*(c1*kH*kW+k*kW+l) + r1*(i*W_out+j) + 1*r2];
                            long ax = bi_data[r0*(c1*kH*kW+k*kW+l) + r1*(i*W_out+j) + 2*r2];
                            assert(c == c1);
                            double px = j+l+offset_data[t0*1+t1*k+t2*l+t3*i+t4*j];
                            double py = i+k+offset_data[t0*0+t1*k+t2*l+t3*i+t4*j];
    

                            
                            double w0, w1, w2, w3;
                            w0 = bw_data[q0*(c1*kH*kW+k*kW+l) + q1*(i*W_out+j) + 0*q2];
                            w1 = bw_data[q0*(c1*kH*kW+k*kW+l) + q1*(i*W_out+j) + 1*q2];
                            w2 = bw_data[q0*(c1*kH*kW+k*kW+l) + q1*(i*W_out+j) + 2*q2];
                            w3 = bw_data[q0*(c1*kH*kW+k*kW+l) + q1*(i*W_out+j) + 3*q2];
                            
                            double px_projected = projection(px,double(W-1));
                            double py_projected = projection(py,double(H-1));
                            assert(w0 == 1 - (py_projected - ay));
                            assert(w1 == 1 - (px_projected - ax));
                            assert(w2 == 1 - ((ay+1)-py_projected));
                            assert(w3 == 1 - ((ax+1)-px_projected));
                            
                            double vx = 0;
                            double vy = 0;
                            vy -= input_data[c1*s0 + ay*s1 + ax*s2]*w1;
                            vy -= input_data[c1*s0 + ay*s1 + (ax+1)*s2]*w3;
                            vy += input_data[c1*s0 + (ay+1)*s1 + ax*s2]*w1;
                            vy += input_data[c1*s0 + (ay+1)*s1 + (ax+1)*s2]*w3;
                            
                            vx -= input_data[c1*s0 + ay*s1 + ax*s2]*w0;
                            vx += input_data[c1*s0 + ay*s1 + (ax+1)*s2]*w0;
                            vx -= input_data[c1*s0 + (ay+1)*s1 + ax*s2]*w2;
                            vx += input_data[c1*s0 + (ay+1)*s1 + (ax+1)*s2]*w2;
                            
                            if(py != py_projected){
                                vy = 0;
                            }
                            
                            if(px != px_projected){
                                vx = 0;
                            }
                            
//                             double epsilon = .00001;
//                             double vx_correct=
// (bilinearInterp(input,NULL,NULL,c1,py,px+epsilon,0,0,0)
// -bilinearInterp(input,NULL,NULL,c1,py,px-epsilon,0,0,0))
// /(2*epsilon);
//                             
//                             double vy_correct =  
// (bilinearInterp(input,NULL,NULL,c1,py+epsilon,px,0,0,0)
// -bilinearInterp(input,NULL,NULL,c1,py-epsilon,px,0,0,0))
// /(2*epsilon);
//                             double err_y = vy_correct - vy;
//                             double err_x = vx_correct - vx;
//                             if(err_x > epsilon or err_y > epsilon){
//                                 std::cout << "the error is " << err_y << " " << err_x << std::endl;
//                                 std::cout << "the location is " << py << " " << px << std::endl;
//                             }
                                   
                            gradOffset_data[(0*kH*kW+k*kW+l)*v0 + i*v1 + j*v2]
                                += gradOutput_data[c2*z0 + i*z1 + j*z2]
                                    *weight_data[c2*u0 + c1*u1 + k*u2 + l*u3]
                                    *vy;
                            gradOffset_data[(1*kH*kW+k*kW+l)*v0 + i*v1 + j*v2]
                                += gradOutput_data[c2*z0 + i*z1 + j*z2]
                                    *weight_data[c2*u0 + c1*u1 + k*u2 + l*u3]
                                    *vx;
                        }
                    }
                }
            }
        }
    }
    
    luaT_pushudata(L, gradOffset, "torch.DoubleTensor");
    return 1;

}  


static const struct luaL_Reg Global_funcs[] = {
	{"im2col", im2col},
    {"bilinearInterpolation", bilinearInterpolation},
    {"update_grad_input", update_grad_input},
    {"grad_offset", grad_offset},
	{NULL, NULL}
};

extern "C"
{

DLL_EXPORT int luaopen_libdeformableconvolution(lua_State* L)
{
	lua_newtable(L);
	luaT_setfuncs(L, Global_funcs, 0);
	lua_setglobal(L, "deformableconvolution");

	return 0;
}

}
