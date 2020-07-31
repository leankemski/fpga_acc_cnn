#include "stdio.h"
#include "conv_core.h"

#define IN_WIDTH 10
#define IN_HEIGHT 10
#define IN_CH 16

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5
#define X_STRIDE 1
#define Y_STRIDE 1

#define RELU_EN  0
#define MODE     0          //0:VALID, 1:SAME
#define X_PADDING (MODE?(KERNEL_WIDTH-1)/2:0)
#define Y_PADDING (MODE?(KERNEL_HEIGHT-1)/2:0)

#define OUT_CH 1
#define OUT_WIDTH ((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1)
#define OUT_HEIGHT ((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1)

void Conv_soft(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,unsigned int mode,unsigned int relu_en,
		float feature_in[],float W[],float bias[],float feature_out[])
{
	int out_width;
    int out_height;

    int pad_x,pad_y;
	if(mode==0)
	{
		pad_x=0;pad_y=0;
	}
	else
	{
		pad_x=(Kx-1)/2;pad_y=(Ky-1)/2;
	}

    out_width=((Win+2*pad_x-Kx)/Sx+1);
    out_height=((Hin+2*pad_y-Ky)/Sy+1);

	//printf("Conv:out_width=%d,out_height=%d\n",out_width,out_height);

	for(int i=0;i<CHout;i++)
		for(int j=0;j<out_height;j++)
			for(int k=0;k<out_width;k++)
			{
				float result=bias[i];
				for(int ki=0;ki<Kx;ki++)
					for(int kj=0;kj<Ky;kj++)
						for(int chi=0;chi<CHin;chi++)
						{
							float data;
							int axis_h=Sy*j+kj-pad_y;
							int axis_w=Sx*k+ki-pad_x;
							if( (axis_h<0) || (axis_h>=Hin) || (axis_w<0) || (axis_w>=Win) )//padding 0
								data=0;
							else
								data=feature_in[axis_h*CHin*Win+axis_w*CHin+chi];
							result+=data*W[kj*CHout*CHin*Kx+ki*CHout*CHin+chi*CHout+i];
						}
				if(relu_en && result<0)
					result=0;
				feature_out[j*CHout*out_width+k*CHout+i]=result;
			}
}

int main(void)
{
	Dtype_f feature_in[IN_HEIGHT][IN_WIDTH][IN_CH];
	Dtype_w W[KERNEL_HEIGHT][KERNEL_WIDTH][IN_CH][OUT_CH];
	Dtype_w bias[OUT_CH];
	Dtype_f feature_out[OUT_HEIGHT][OUT_WIDTH][OUT_CH];
	Dtype_f feature_out_soft[OUT_HEIGHT][OUT_WIDTH][OUT_CH];

	for(int i=0;i<IN_HEIGHT;i++)
		for(int j=0;j<IN_WIDTH;j++)
			for(int cin=0;cin<IN_CH;cin++)
				feature_in[i][j][cin]=i*IN_WIDTH+j;

	for(int i=0;i<KERNEL_HEIGHT;i++)
		for(int j=0;j<KERNEL_WIDTH;j++)
			for(int cin=0;cin<IN_CH;cin++)
				for(int cout=0;cout<OUT_CH;cout++)
					W[i][j][cin][cout]=i*KERNEL_WIDTH+j;

	for(int cout=0;cout<OUT_CH;cout++)
		bias[cout]=0;

	printf("start\n");

	Conv(IN_CH,IN_HEIGHT,IN_WIDTH,OUT_CH,
			KERNEL_WIDTH,KERNEL_HEIGHT,X_STRIDE,Y_STRIDE,MODE,RELU_EN,
			feature_in[0][0],W[0][0][0],bias,feature_out[0][0]
		);

	Conv_soft(IN_CH,IN_HEIGHT,IN_WIDTH,OUT_CH,
				KERNEL_WIDTH,KERNEL_HEIGHT,X_STRIDE,Y_STRIDE,MODE,RELU_EN,
				feature_in[0][0],W[0][0][0],bias,feature_out_soft[0][0]
			);

	for(int i=0;i<OUT_HEIGHT;i++)
		for(int j=0;j<OUT_WIDTH;j++)
			for(int cout=0;cout<OUT_CH;cout++)
			{
				if(feature_out[i][j][cout] != feature_out_soft[i][j][cout])
					printf("OUT[%d][%d][%d]=%f\nOUT_SOFT[%d][%d][%d]=%f\n",i,j,cout,feature_out[i][j][cout],i,j,cout,feature_out_soft[i][j][cout]);
			}
	printf("done!\n");

	return 0;
}
