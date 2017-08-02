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
	write_model(model_path);
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
		else{
			printf("unknown layer: %s\n",layer_type.c_str());
			exit(0);
		}
	}
	ofile<<"Connections:"<<endl;
	for(auto i:connections) ofile<<i.first<<" "<<i.second<<endl;
	ofile.close();
}
void CaffeModelParser::write_net_conv(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile){
	const caffe::ConvolutionParameter& conv_param = param.convolution_param();
	ofile<<"Layer: conv "<<layer_name<<endl;
	ofile<<conv_param.kernel_size()[0]<<" "<<conv_param.num_output()<<" "<<conv_param.pad()[0]<<" "<<conv_param.stride()[0]<<" none"<<endl;
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
	ofile<<pool_param.kernel_size()<<" "<<pool_param.pad()<<" "<<pool_param.stride()<<" "<<method<<endl;
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
