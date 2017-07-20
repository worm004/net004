#include "stdlib.h"
#include "FCLayer.h"

FCLayer::FCLayer(
	const std::string&name, 
	int n, 
	const std::string& activity):
		n(n), 
		activity(activity),
		Layer(name,"fc"){
}
FCLayer::~FCLayer(){
}
void FCLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());

	//printf("input:\n");
	//for(int i=0;i<std::min(100,inputs[0].total());++i)
	//	printf("%f ",inputs[0].data[i]);
		
	float * idata = inputs[0].data,
		* odata = outputs[0].data;
	
	int w = inputs[0].c * inputs[0].h * inputs[0].w,
	    h = n,
	    batch_size = inputs[0].n;

	//for(int b=0;b<2;++b){
	//	printf("\ninput %d:\n",b);
	//	for(int k=0;k<1;++k){
	//		for(int i=0;i<inputs[0].h;++i){
	//			for(int j=0;j<inputs[0].w;++j)
	//				printf("%f ",inputs[0].data[b*inputs[0].h*inputs[0].w*inputs[0].c + inputs[0].h * inputs[0].w *k + i*inputs[0].w + j]);
	//			printf("\n");
	//		}
	//		printf("\n");
	//	}
	//}

	for(int b = 0; b < batch_size; ++b){
		float *weight_data = weight.data;
		for(int y = 0; y < h; ++y){
			float val = 0.0f;
			for(int x=0;x<w;++x)
				val += idata[x] * weight_data[x];
			odata[y] = val + bias.data[y];
			weight_data += w;
		}
		odata += h;
		idata += w;
	}

	//for(int b=0;b<2;++b){
	//	printf("\noutput %d:\n",b);
	//	for(int k=0;k<1;++k){
	//		for(int i=0;i<1/*outputs[0].h*/;++i){
	//			for(int j=0;j<outputs[0].w;++j)
	//				printf("%f ",outputs[0].data[b*outputs[0].h*outputs[0].w*outputs[0].c + outputs[0].h * outputs[0].w *k + i*outputs[0].w + j]);
	//		}
	//	}
	//	printf("\n");
	//}
	//getchar();
		
	//for(int b = 0; b < batch_size; ++b){
	//	for(int o = 0; o < on; ++o){
	//		float *weight_data = weight.data + o*in;
	//		float val = 0.0f;
	//		for(int i=0;i<in;++i)
	//			val += idata[i] * weight_data[i];
	//		odata[o] = val + bias.data[o];
	//	}
	//	odata += on;
	//	idata += inputs[0].c * inputs[0].h * inputs[0].w;
	//}

	//for(int i=0;i<n;++i){
	//	float val = 0.0f;
	//	for(int j=0;j<in;++j){
	//		val += data[j] * weight_data[j+i*in];
	//	}
	//	odata[i] = val + bias.data[i];
	//}

	//printf("\noutput:\n");
	//for(int i=0;i<std::min(100,outputs[0].total());++i)
	//	printf("%f ",outputs[0].data[i]);
	//getchar();
}
void FCLayer::backward(){
}
void FCLayer::show()const {
	printf("[%s%s] name: %s, n: %d\n",
			type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
			name.c_str(),n);

	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(bias.total() != 0){
		printf("\tbias: ");
		bias.show();
	}
	if(weight.total() != 0){
		printf("\tweight: ");
		weight.show();
	}

	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}

}
void FCLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: fc input blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	const Blob& ib = inputs[0];
	int in = ib.c*ib.h*ib.h;

	weight.set_shape(1,in * n , 1, 1);
	weight_dif.set_shape(weight);
	bias.set_shape(1,n,1,1);
	bias_dif.set_shape(bias);

	// output
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(ib.n, n, 1, 1);
	output_difs[0].set_shape(outputs[0]);
}
void FCLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: fc output blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	weight.alloc();
	weight_dif.alloc();
	bias.alloc();
	bias_dif.alloc();

	// output
	outputs[0].alloc();
	output_difs[0].alloc();
}
int FCLayer::parameter_number(){
	return weight.total() + bias.total();
}
