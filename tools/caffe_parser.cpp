#include <map>
#include "caffe_parser.h"
using namespace std;
using namespace cv;
void CaffeModelParser::load_caffe_model(const std::string& net_path, const std::string& model_path){
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	read_connections();
}
void CaffeModelParser::read_connections(){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<string> layer_names = net->layer_names();
	const vector<string> blob_names = net->blob_names();

	vector<pair<string,string>> tops;
	for(int i=0;i<layers.size();++i){
		string layer_name = layer_names[i], layer_type = layers[i]->type();
		const vector<int> bottom_ids = net->bottom_ids(i);
		// connect
		for(int j=0;j<bottom_ids.size();++j){
			for(int k=0;k<tops.size();){
				string bname = blob_names[bottom_ids[j]];
				if(tops[k].second == bname){
					connections.push_back(make_pair(tops[k].first,layer_name));
					tops.erase(tops.begin()+k);
				} else ++k;
			}
		}

		// add new
		const vector<int> top_ids = net->top_ids(i);
		for(int j=0;j<top_ids.size();++j){
			if(layer_type == "Input")
				tops.push_back(make_pair(blob_names[top_ids[j]], blob_names[top_ids[j]]));
			else tops.push_back(make_pair(layer_name, blob_names[top_ids[j]]));
		}
	}
	//for(auto i:connections){
	//	printf("%s -> %s\n",i.first.c_str(),i.second.c_str());
	//}
}
void CaffeModelParser::write(const std::string& net_path, const std::string& model_path){
	write_net(net_path);
	//write_model(model_path);
	write_model2(model_path);
}

void CaffeModelParser::write_blob(const std::string& layer_name, const std::string& blob_name, const caffe::Blob<float> *blob, FILE* file){
	char buffer[100];
	sprintf(buffer,"Layer: %s %s",layer_name.c_str(),blob_name.c_str());
	fwrite(buffer, sizeof(char), 100, file);
	int num = blob->num(), channel = blob->channels(), height = blob->height(), width = blob->width();
	int total = num * channel * height * width;
	fwrite(&num, sizeof(int), 1, file);
	fwrite(&channel, sizeof(int), 1, file);
	fwrite(&height, sizeof(int), 1, file);
	fwrite(&width, sizeof(int), 1, file);
	fwrite(blob->cpu_data(), sizeof(float), total, file);
}
void CaffeModelParser::write_model2(const std::string& model_path){
	FILE* file = fopen(model_path.c_str(), "wb");
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<string>& layer_names = net->layer_names(), &blob_names = net->blob_names();
	for(int i=0;i<layers.size();++i){
		const vector<boost::shared_ptr<caffe::Blob<float> > >& params = layers[i]->blobs();
		const string& layer_type = layers[i]->type();
		if((layer_type == "Convolution") || (layer_type == "InnerProduct")){
			write_blob(layer_names[i], "weight", params[0].get(),file);
			if(params.size()>1) write_blob(layer_names[i], "bias", params[1].get(),file);
		} 
		else if(layer_type == "BatchNorm"){
			write_blob(layer_names[i], "mean", params[0].get(),file);
			if(params.size()>1) write_blob(layer_names[i], "variance", params[1].get(),file);
			if(params.size()>2) write_blob(layer_names[i], "scale", params[2].get(),file);
		}
		else if(layer_type == "Scale"){
			write_blob(layer_names[i], "scale", params[0].get(),file);
			if(params.size()>1) write_blob(layer_names[i], "bias", params[1].get(),file);
		}
	}
	fclose(file);
}
void CaffeModelParser::write_model(const std::string& model_path){
	ofstream ofile(model_path);
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<string>& layer_names = net->layer_names(), &blob_names = net->blob_names();
	for(int i=0;i<layers.size();++i){
		const vector<boost::shared_ptr<caffe::Blob<float> > >& params = layers[i]->blobs();
		const string& layer_type = layers[i]->type();
		if((layer_type == "Convolution") || (layer_type == "InnerProduct")){
			caffe::Blob<float> *bweight = params[0].get();
			ofile<<"Layer: "<<layer_names[i]<<" weight"<<endl;
			ofile<<bweight->num()<<" "<<bweight->channels()<<" "<<bweight->height()<<" "<<bweight->width()<<endl;
			int total = bweight->num() * bweight->channels() * bweight->height() * bweight->width();
			const float *data = bweight->cpu_data();
			for(int j=0;j<total;++j) ofile<<data[j]<<" ";
			ofile<<endl;

			caffe::Blob<float> *bbias = params[1].get();
			ofile<<"Layer: "<<layer_names[i]<<" bias"<<endl;
			ofile<<bbias->num() <<" "<< bbias->channels() <<" "<< bbias->height() <<" "<< bbias->width()<<endl;
			total = bbias->num() * bbias->channels() * bbias->height() * bbias->width();
			data = bbias->cpu_data();
			for(int j=0;j<total;++j) ofile<<data[j]<<" ";
			ofile<<endl;
		}
	}
}
void CaffeModelParser::write_net(const std::string& net_path){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<string>& layer_names = net->layer_names(), &blob_names = net->blob_names();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();
	ofstream ofile(net_path);
	ofile<<net->name()<<endl;
	for(int i=0;i<layers.size();++i){
		string layer_type = layers[i]->type(), 
		       layer_name = layer_names[i];
		//printf("%d %s %s\n",i,layer_type.c_str(),layer_name.c_str());
		const caffe::LayerParameter& param = layers[i]->layer_param();
		if(layer_type == "Input"){
			const vector<int> top_ids = net->top_ids(i);
			for(int j=0;j<top_ids.size();++j){
				caffe::Blob<float> *b = tops[i][j];
				write_net_data(layer_name, blob_names[top_ids[j]], 
						b->num(), b->channels(), b->height(), b->width(), ofile);
			}
		}
		else if(layer_type == "Convolution") write_net_conv(layer_name, param, ofile);
		else if(layer_type == "Pooling") write_net_pool(layer_name, param, ofile);
		else if(layer_type == "ReLU") write_net_relu(layer_name, param, ofile);
		else if(layer_type == "InnerProduct") write_net_fc(layer_name, param, ofile);
		else if(layer_type == "SoftmaxWithLoss") write_net_softmaxloss(layer_name, param, ofile);
		else if(layer_type == "LRN") write_net_lrn(layer_name, param, ofile);
		else if(layer_type == "Split") write_net_split(layer_name, ofile);
		else if(layer_type == "Concat") write_net_concat(layer_name, ofile);
		else if(layer_type == "BatchNorm") write_net_bn(layer_name,param,ofile);
		else if(layer_type == "Scale") write_net_scale(layer_name,param,ofile);
		else if(layer_type == "Eltwise") write_net_eltwise(layer_name,param,ofile);
		else{
			printf("unknown layer: %s\n",layer_type.c_str());
			exit(0);
		}
	}
	ofile<<"Connections:"<<endl;
	for(auto i:connections) ofile<<i.first<<" "<<i.second<<endl;
	ofile.close();
}
void CaffeModelParser::write_net_eltwise(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	ofile<<"Layer: eltwise "<<layer_name<<endl;
	const caffe::EltwiseParameter& eltwise_param = param.eltwise_param();
	if(eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_SUM){
		ofile<<"sum"<<endl;
	}
	else if(eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_MAX){
		ofile<<"max"<<endl;
	}
	else if(eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_PROD){
		ofile<<"prod"<<endl;
	}
	else{
		printf("no such op in eltwise\n");
		exit(0);
	}
}
void CaffeModelParser::write_net_scale(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	ofile<<"Layer: scale "<<layer_name<<endl;
	ofile<<endl;
}
void CaffeModelParser::write_net_bn(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	ofile<<"Layer: batchnorm "<<layer_name<<endl;
	ofile<<endl;
}
void CaffeModelParser::write_net_concat(const std::string& layer_name, std::ofstream& ofile){
	ofile<<"Layer: concat "<<layer_name<<endl;
	ofile<<"channel"<<endl;
}
void CaffeModelParser::write_net_split(const std::string& layer_name, std::ofstream& ofile){
	ofile<<"Layer: split "<<layer_name<<endl;
	ofile<<endl;
}
void CaffeModelParser::write_net_conv(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::ConvolutionParameter& conv_param = param.convolution_param();
	ofile<<"Layer: conv "<<layer_name<<endl;
	int kernel_size = conv_param.kernel_size().size() == 1?conv_param.kernel_size()[0]:1;
	int pad = conv_param.pad().size() == 1? conv_param.pad()[0]:0;
	int stride = conv_param.stride().size() == 1? conv_param.stride()[0]:1;
	int group = conv_param.group();

	ofile<<kernel_size<<" "<<conv_param.num_output()<<" "<<pad<<" "<<stride<<" "<<group<<" none"<<endl;
}
void CaffeModelParser::write_net_pool(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::PoolingParameter& pool_param = param.pooling_param();
	ofile<<"Layer: pool "<<layer_name<<endl;
	string method;
	if(pool_param.pool() == caffe::PoolingParameter_PoolMethod_MAX)
		method = "max";
	else if(pool_param.pool() == caffe::PoolingParameter_PoolMethod_AVE)
		method = "avg";
	else{
		printf("should not touch here\n");
		exit(0);
	}
	int k = pool_param.global_pooling()?-1:pool_param.kernel_size();
	ofile<<k<<" "<<pool_param.pad()<<" "<<pool_param.stride()<<" "<<method<<endl;
}
void CaffeModelParser::write_net_data(const std::string& layer_name, const std::string& blob_name, int n,int c,int h,int w, std::ofstream& ofile){
	ofile<<"Layer: data "<<blob_name<<endl;
	string method;
	if(blob_name.find("label") != string::npos){
		method = "label";
	}else method = "image";
	ofile<<n<<" "<<c<<" "<<h<<" "<<w<<" "<<method<<endl;
}
void CaffeModelParser::write_net_relu(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::ReLUParameter& relu_param = param.relu_param();
	ofile<<"Layer: activity "<<layer_name<<endl;
	ofile<<"relu"<<endl;
}
void CaffeModelParser::write_net_fc(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::InnerProductParameter& fc_param = param.inner_product_param();
	ofile<<"Layer: fc "<<layer_name<<endl;
	ofile<<fc_param.num_output()<<" none"<<endl;
}
void CaffeModelParser::write_net_softmaxloss(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	ofile<<"Layer: loss "<<layer_name<<endl;
	ofile<<"softmax"<<endl;
}
void CaffeModelParser::write_net_lrn(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::LRNParameter& lrn_param = param.lrn_param();
	ofile<<"Layer: lrn "<<layer_name<<endl;
	ofile<<lrn_param.local_size()<<" "<<lrn_param.alpha()<<" "<<lrn_param.beta()<<endl;

}
void CaffeModelParser::show_layers(){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<string> layer_names = net->layer_names();
	const vector<string> blob_names = net->blob_names();
	for(int i=0;i<layers.size();++i){
		printf("Layers: %s %s\n",layers[i]->type(), layer_names[i].c_str());
		const vector<int> top_ids = net->top_ids(i);
		const vector<int> bottom_ids = net->bottom_ids(i);
		for(int j=0;j<bottom_ids.size();++j)
			printf("bottom Blob: %s\n",blob_names[bottom_ids[j]].c_str());
		for(int j=0;j<top_ids.size();++j)
			printf("top Blob: %s\n",blob_names[top_ids[j]].c_str());
	}
}
