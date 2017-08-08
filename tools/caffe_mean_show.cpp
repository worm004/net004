#include "caffe/util/io.hpp"
using namespace caffe;

int main(){
	BlobProto mean_blob;
	ReadProtoFromBinaryFile("../caffe_models/ResNet_mean.binaryproto",&mean_blob);
	int c = mean_blob.channels();
	int h = mean_blob.height();
	int w = mean_blob.width();
	printf("%d %d %d\n",c,h,w);
	for(int i=0;i<c;++i){
		for(int j=0;j<h*w;++j)
			printf(" %g",mean_blob.data(i*h*w+j));
		printf("\n");
	}

	return 0;
}
