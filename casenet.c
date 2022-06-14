#include<ap_cint.h>
//#include <ap_fixed.h>
#include<math.h>



// typedef ap_fixed<16,11> fixed16;

#define IN_DATASIZE 196
#define OUT_DATASIZE 3

#define C1_OCH 64
#define C1_OSIZE 36
#define C1_ICH 1
#define C1_ISIZE 40
#define C1_K 5

#define P1_OSIZE 18
#define P1_K 2

#define C2_OCH 128
#define C2_OSIZE 16
#define C2_ICH 64
#define C2_ISIZE 18
#define C2_K 3
#define C2_P 10

#define P2_OSIZE 8
#define P2_K 2

#define C3_OCH 64
#define C3_OSIZE 6
#define C3_ICH 128
#define C3_ISIZE 8
#define C3_K 3
#define C3_P 10

#define P3_OSIZE 3
#define P3_K 2

#define F1_N 128
#define F1_M 576
#define F1_P 10

#define F2_N 10
#define F2_M 128
#define F2_P 10

#define RESULTSIZE 12

//Old number of weights: 431080
//New number of weights: 299594
//64 32 64: 1664 + 18464 + 18496 + 57700 + 1010 = 97334
//32 64 32 100: 832 + 18496 + 18464 + 28900 + 1010 = 67702
//32 64 32 256: 832 + 18496 + 18464 + 73984 + 2570 = 114346
//32 32 32 100: 832 + 9248 + 9248 + 28900 + 1010 = 49238
//64 128 64 128: 1664 + 73856 + 73792 + 73856 + 1408 = 224576
#define ALL_WB_SIZE (1664 + 73856 + 73792 + 73856 + 1408)

void load_input(
		float input[C1_ICH*C1_ISIZE*C1_ISIZE],
		float output[C1_ICH][C1_ISIZE][C1_ISIZE]
		){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
//#pragma HLS array_partition variable=tmp complete
	int i, j, k;
	float tmp[C1_ICH*C1_ISIZE*C1_ISIZE];

//#pragma HLS UNROLL
	for (i = 0; i < C1_ICH*C1_ISIZE*C1_ISIZE; i++) {
		tmp[i] = input[i];
	}
//#pragma HLS PIPELINE
	for(i = 0; i < C1_ICH; i++) {
		for(j = 0; j < C1_ISIZE; j++) {
//#pragma HLS PIPELINE
			for(k = 0; k < C1_ISIZE; k++) {
				output[i][j][k] = tmp[i*C1_ISIZE*C1_ISIZE + j*C1_ISIZE + k];
			}
		}
	}
	return;
}

void load_wb(
		float input[ALL_WB_SIZE],
		float conv1_w[C1_OCH][C1_ICH][C1_K][C1_K],
		float conv1_b[C1_OCH],
		float conv2_w[C2_OCH][C2_ICH][C2_K][C2_K],
		float conv2_b[C2_OCH],
		float conv3_w[C3_OCH][C3_ICH][C3_K][C3_K],
		float conv3_b[C3_OCH],
		float fc1_w[F1_N][F1_M],
		float fc1_b[F1_N],
		float fc2_w[F2_N][F2_M],
		float fc2_b[F2_N]
		){
#pragma HLS INLINE off
	int i, j, k, l;
	unsigned long datasize, offset;
	//datasize = (unsigned long)input[0];
	//printf("HLS: datasize=%lu\n", datasize);

//	float tmp[ALL_WB_SIZE];
//
//	for (i = 0; i < ALL_WB_SIZE; i++) {
//		tmp[i] = input[i];
//	}

	//CONV1_WB
	offset = 0;
	for(i = 0; i < C1_OCH; i++) {
		for(j = 0; j < C1_ICH; j++) {
			for(k = 0; k < C1_K; k++) {
				for (l = 0; l < C1_K; l++) {
					conv1_w[i][j][k][l] = input[offset + i*C1_ICH*C1_K*C1_K + j*C1_K*C1_K + k*C1_K + l];
				}
			}
		}
	}
	for(i = 0; i < C1_OCH; i++) {
		conv1_b[i] = input[offset + C1_OCH*C1_ICH*C1_K*C1_K + i];
	}

	//CONV2_WB
	offset = 1664;
	for(i = 0; i < C2_OCH; i++) {
		for(j = 0; j < C2_ICH; j++) {
			for(k = 0; k < C2_K; k++) {
				for (l = 0; l < C2_K; l++) {
					conv2_w[i][j][k][l] = input[offset + i*C2_ICH*C2_K*C2_K + j*C2_K*C2_K + k*C2_K + l];
				}
			}
		}
	}
	for(i = 0; i < C2_OCH; i++) {
		conv2_b[i] = input[offset + C2_OCH*C2_ICH*C2_K*C2_K + i];
	}

	//CONV3_WB
	offset = 1664 + 73856;
	for(i = 0; i < C3_OCH; i++) {
		for(j = 0; j < C3_ICH; j++) {
			for(k = 0; k < C3_K; k++) {
				for (l = 0; l < C3_K; l++) {
					conv3_w[i][j][k][l] = input[offset + i*C3_ICH*C3_K*C3_K + j*C3_K*C3_K + k*C3_K + l];
				}
			}
		}
	}
	for(i = 0; i < C3_OCH; i++) {
		conv3_b[i] = input[offset + C3_OCH*C3_ICH*C3_K*C3_K + i];
	}

	//FC1_WB
	offset = 1664 + 73856 + 73792;
	for(i = 0; i < F1_N; i++)
		for (j = 0; j < F1_M; j++)
			fc1_w[i][j] = input[offset + i*F1_M + j];
	for(i = 0; i < F1_N; i++)
		fc1_b[i] = input[offset + F1_N*F1_M + i];

	//FC1_WB
	offset = 1664 + 73856 + 73792 + 73856;
	for(i = 0; i < F2_N; i++)
		for (j = 0; j < F2_M; j++)
			fc2_w[i][j] = input[offset + i*F2_M + j];
	for(i = 0; i < F2_N; i++)
		fc2_b[i] = input[offset + F2_N*F2_M + i];

	return;
}

void conv1(
		float input[C1_ICH][C1_ISIZE][C1_ISIZE],
		float weight[C1_OCH][C1_ICH][C1_K][C1_K],
		float bias[C1_OCH],
		float output[C1_OCH][C1_OSIZE][C1_OSIZE]
){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
//#pragma HLS array_partition variable=weight complete
//#pragma HLS PIPELINE
	int ox, oy, kx, ky, n, m;
	static int stride = 1;

	//Calculate
		for (ox = 0; ox < C1_OSIZE; ox++) {
			for (oy = 0; oy < C1_OSIZE; oy++) {
//#pragma HLS PIPELINE
				for (n = 0; n < C1_OCH; n++) {
					output[n][ox][oy] = bias[n];
//#pragma HLS PIPELINE
					for (m = 0; m < C1_ICH; m++) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
						for (kx = 0; kx < C1_K; kx++) {
							for (ky = 0; ky < C1_K; ky++) {

								//ix[kx] = stride*ox+kx;
								//iy[ky] = stride*oy+ky;
								output[n][ox][oy] +=
										weight[n][m][kx][ky] *
										input[m][stride*ox+kx][stride*oy+ky];

							}
						}
					}
					if (output[n][ox][oy]<0)
						output[n][ox][oy]=0;
				}
			}
		}
	return;
}

void pool1(
		float input[C1_OCH][C1_OSIZE][C1_OSIZE],
		float output[C1_OCH][P1_OSIZE][P1_OSIZE]
		){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
	int ox, oy, kx, ky, ix, iy, n, m;
	float tmp, max;

	int stride = 2;
	  for (n = 0; n < C1_OCH; n++) {
//#pragma HLS PIPELINE
		for (ox = 0; ox < P1_OSIZE; ox++) {
//#pragma HLS PIPELINE
		  for (oy = 0; oy < P1_OSIZE; oy++) {
			max = -256.0;
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
			for (kx = 0; kx < P1_K; kx++) {
			  for (ky = 0; ky < P1_K; ky++) {
				 tmp = input[n][ox*stride+kx][oy*stride+ky];
				//tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
				if (max < tmp) max = tmp;
			  }
			}
			output[n][ox][oy] = max;
			//*(output+och*osize*osize+osize*orow+ocol) = max;
		  }
		}
	  }
	return;
}

void conv2(
		float input[C2_ICH][C2_ISIZE][C2_ISIZE],
		float weight[C2_OCH][C2_ICH][C2_K][C2_K],
		float bias[C2_OCH],
		float output[C2_OCH][C2_OSIZE][C2_OSIZE]
){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
//#pragma HLS array_partition variable=weight complete
//#pragma HLS PIPELINE
//#pragma HLS array_partition variable=bias complete
	int ox, oy, kx, ky, n, m;
	static int stride = 1;

	//Calculate
		for (ox = 0; ox < C2_OSIZE; ox++) {
			for (oy = 0; oy < C2_OSIZE; oy++) {
//#pragma HLS PIPELINE
				for (n = 0; n < C2_OCH; n++) {
//#pragma HLS PIPELINE
					output[n][ox][oy] = bias[n];
					for (m = 0; m < C2_ICH; m++) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
						for (kx = 0; kx < C2_K; kx++) {
							for (ky = 0; ky < C2_K; ky++) {
								//ix[kx] = stride*ox+kx;
								//iy[ky] = stride*oy+ky;
								output[n][ox][oy] +=
										weight[n][m][kx][ky] *
										input[m][stride*ox+kx][stride*oy+ky];
							}
						}
					}
					if (output[n][ox][oy]<0)
						output[n][ox][oy]=0;
				}
			}
		}
	return;
}

void pool2(
		float input[C2_OCH][C2_OSIZE][C2_OSIZE],
		float output[C2_OCH][P2_OSIZE][P2_OSIZE]
		){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
	int ox, oy, kx, ky, ix, iy, n, m;
	float tmp, max;

	int stride = 2;
	  for (n = 0; n < C2_OCH; n++) {
//#pragma HLS PIPELINE
		for (ox = 0; ox < P2_OSIZE; ox++) {
//#pragma HLS PIPELINE
		  for (oy = 0; oy < P2_OSIZE; oy++) {
			max = -256.0;
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
			for (kx = 0; kx < P2_K; kx++) {
			  for (ky = 0; ky < P2_K; ky++) {
				 tmp = input[n][ox*stride+kx][oy*stride+ky];
				//tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
				if (max < tmp) max = tmp;
			  }
			}
			output[n][ox][oy] = max;
			//*(output+och*osize*osize+osize*orow+ocol) = max;
		  }
		}
	  }
	return;
}

void conv3(
		float input[C3_ICH][C3_ISIZE][C3_ISIZE],
		float weight[C3_OCH][C3_ICH][C3_K][C3_K],
		float bias[C3_OCH],
		float output[C3_OCH][C3_OSIZE][C3_OSIZE]
){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
//#pragma HLS array_partition variable=weight complete
//#pragma HLS PIPELINE
	int ox, oy, kx, ky, n, m;
	static int stride = 1;

	//Calculate
		for (ox = 0; ox < C3_OSIZE; ox++) {
			for (oy = 0; oy < C3_OSIZE; oy++) {
//#pragma HLS PIPELINE
				for (n = 0; n < C3_OCH; n++) {
//#pragma HLS PIPELINE
					output[n][ox][oy] = bias[n];
					for (m = 0; m < C3_ICH; m++) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
						for (kx = 0; kx < C3_K; kx++) {
							for (ky = 0; ky < C3_K; ky++) {

								//ix[kx] = stride*ox+kx;
								//iy[ky] = stride*oy+ky;
								output[n][ox][oy] +=
										weight[n][m][kx][ky] *
										input[m][stride*ox+kx][stride*oy+ky];

							}
						}
					}
					if (output[n][ox][oy]<0)
						output[n][ox][oy]=0;
				}
			}
		}
	return;
}

void pool3(
		float input[C3_OCH][C3_OSIZE][C3_OSIZE],
		float output[C3_OCH][P3_OSIZE][P3_OSIZE]
		){
#pragma HLS INLINE off
//#pragma HLS array_partition variable=input complete
	int ox, oy, kx, ky, ix, iy, n, m;
	float tmp, max;

	int stride = 2;
//#pragma HLS PIPELINE
	  for (n = 0; n < C3_OCH; n++) {
//#pragma HLS PIPELINE
		  for (ox = 0; ox < P3_OSIZE; ox++) {
		  for (oy = 0; oy < P3_OSIZE; oy++) {
//pragma HLS UNROLL
//#pragma HLS PIPELINE
			max = -256.0;
			for (kx = 0; kx < P3_K; kx++) {
			  for (ky = 0; ky < P3_K; ky++) {
				 tmp = input[n][ox*stride+kx][oy*stride+ky];
				//tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
				if (max < tmp) max = tmp;
			  }
			}
			output[n][ox][oy] = max;
			//*(output+och*osize*osize+osize*orow+ocol) = max;
		  }
		}
	  }
	return;
}

void flatten(float input[C3_OCH][P3_OSIZE][P3_OSIZE], float output[F1_M]){
//#pragma HLS INLINE off
//#pragma HLS array_partition variable=input cyclic factor=4 dim=1
	int ox, oy, n;
//#pragma HLS PIPELINE
	for (n = 0; n < C3_OCH; n++) {
//#pragma HLS PIPELINE
		for (ox = 0; ox < P3_OSIZE; ox++) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
		 	for (oy = 0; oy < P3_OSIZE; oy++) {

		 		output[n*P3_OSIZE*P3_OSIZE + ox*P3_OSIZE + oy] = input[n][ox][oy];
		 	}
		}
	}
	return;
}

void fc1(float input[F1_M], float weight[F1_N][F1_M], float bias[F1_N], float output[F1_N]) {
#pragma HLS INLINE off
//#pragma HLS array_partition variable=weight cyclic factor=4 dim=1
//#pragma HLS array_partition variable=input cyclic factor=4 dim=1
//#pragma HLS array_partition variable=bias cyclic factor=4 dim=1
	int i, j, p;
//#pragma HLS PIPELINE
	for (i = 0; i < F1_N; i++) {
		output[i] = bias[i];
//#pragma HLS PIPELINE
		for (j = 0; j < F1_M; j+=F1_P) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
			for (p = 0; p < F1_P; p++) {
				output[i] += input[j+p] * weight[i][j+p];
			}
		}
		if (output[i] < 0.0) output[i] = 0.0;
	}
	return;
}

void fc2(float input[F2_M], float weight[F2_N][F2_M], float bias[F2_N], float output[F2_N]) {
//#pragma HLS array_partition variable=weight cyclic factor=4 dim=1
//#pragma HLS array_partition variable=input cyclic factor=4 dim=1
#pragma HLS INLINE off
	int i, j, k, l, p;
	float sum = 0.0;
	float output_tmp[F2_N];
//#pragma HLS PIPELINE
	for (i = 0; i < F2_N; i++) {
		output_tmp[i] = bias[i];
//#pragma HLS PIPELINE
		for (j = 0; j < F2_M; j+=F2_P) {
//#pragma HLS UNROLL
//#pragma HLS PIPELINE
			for (p = 0; p < F2_P; p++) {
				output_tmp[i] += input[j+p] * weight[i][j+p];
			}
		}
	}
	for (k = 0; k < F2_N; k++) {
		sum += expf(output_tmp[k]);
	}
	for (l = 0; l < F2_N; l++) {
		output[l] = expf(output_tmp[l]) / sum;
	}
	return;
}

void store_output(float input[F2_N], float output[RESULTSIZE]){
#pragma HLS INLINE off
	int i;
	for(i = 0; i < F2_N; i++)
		output[i] = input[i];
	for(i = F2_N; i < RESULTSIZE; i++)
		output[i] = 0.0;
	return;
}

void lenetcon(
		float input[C1_ICH*C1_ISIZE*C1_ISIZE],
		float output[RESULTSIZE],
		float wb[ALL_WB_SIZE]
){
//#pragma HLS dataflow
#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=wb
#pragma HLS INTERFACE s_axilite port=return

	//static float input_tmp[C1_ICH][C1_ISIZE][C1_ISIZE];

	static float image[C1_ICH][C1_ISIZE][C1_ISIZE];

	static float conv1_w[C1_OCH][C1_ICH][C1_K][C1_K];
//#pragma HLS RESOURCE variable=conv1_w core=XPM_MEMORY uram

	static float conv1_b[C1_OCH];
	static float conv1_out[C1_OCH][C1_OSIZE][C1_OSIZE];

	static float pool1_out[C1_OCH][P1_OSIZE][P1_OSIZE];

	static float conv2_w[C2_OCH][C2_ICH][C2_K][C2_K];
//#pragma HLS RESOURCE variable=conv2_w core=XPM_MEMORY uram

	static float conv2_b[C2_OCH];
	static float conv2_out[C2_OCH][C2_OSIZE][C2_OSIZE];

	static float pool2_out[C2_OCH][P2_OSIZE][P2_OSIZE];

	static float conv3_w[C3_OCH][C3_ICH][C3_K][C3_K];
//#pragma HLS RESOURCE variable=conv3_w core=XPM_MEMORY uram

	static float conv3_b[C3_OCH];
	static float conv3_out[C3_OCH][C3_OSIZE][C3_OSIZE];

	static float pool3_out[C3_OCH][P3_OSIZE][P3_OSIZE];

	static float flat_out[F1_M];

	static float fc1_w[F1_N][F1_M];
//#pragma HLS RESOURCE variable=fc1_w core=XPM_MEMORY uram

	static float fc1_b[F1_N];
	static float fc1_out[F1_N];

	static float fc2_w[F2_N][F2_M];
//#pragma HLS RESOURCE variable=fc2_w core=XPM_MEMORY uram
	static float fc2_b[F2_N];
	static float fc2_out[F2_N];

	static int wb_flag = 0;

	if (wb_flag == 0) {
		load_wb(wb, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, fc1_w, fc1_b, fc2_w, fc2_b);
		wb_flag = 1;
	}

	load_input(input, image);

	conv1(image, conv1_w, conv1_b, conv1_out);
	pool1(conv1_out, pool1_out);

	conv2(pool1_out, conv2_w, conv2_b, conv2_out);
	pool2(conv2_out, pool2_out);

	conv3(pool2_out, conv3_w, conv3_b, conv3_out);
	pool3(conv3_out, pool3_out);

	flatten(pool3_out, flat_out);

	fc1(flat_out, fc1_w, fc1_b, fc1_out);

	fc2(fc1_out, fc2_w, fc2_b, fc2_out);

	store_output(fc2_out, output);

	return;
}
