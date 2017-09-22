#include "stdlib.h"
#include <map>
#include "caffe_parser.h"
using namespace std;
CaffeParser::CaffeParser(){
	caffe_param_table = {
		{"Convolution", {"weight","bias"}},
		{"InnerProduct", {"weight","bias"}},
		{"BatchNorm", {"mean","variance","scale"}},
		{"Scale", {"scale","bias"}}
		};
	find_attrs_funcs = {
		{"Convolution", &CaffeParser::find_conv_attrs},
		{"Pooling", &CaffeParser::find_pooling_attrs},
		{"ReLU", &CaffeParser::find_relu_attrs},
		{"InnerProduct", &CaffeParser::find_fc_attrs},
		{"SoftmaxWithLoss", &CaffeParser::find_softmaxloss_attrs},
		{"LRN", &CaffeParser::find_lrn_attrs},
		{"Split", &CaffeParser::find_split_attrs},
		{"Concat", &CaffeParser::find_concat_attrs},
		{"BatchNorm", &CaffeParser::find_bn_attrs},
		{"Scale", &CaffeParser::find_scale_attrs},
		{"Eltwise", &CaffeParser::find_eltwise_attrs},
		{"Reshape", &CaffeParser::find_reshape_attrs},
		{"Softmax", &CaffeParser::find_softmax_attrs},
		{"Python", &CaffeParser::find_python_attrs},
		{"ROIPooling", &CaffeParser::find_roipooling_attrs}
		};
}
void CaffeParser::load_caffe_model(
	const std::string& net_path, 
	const std::string& model_path, 
	bool is_train){
	this->is_train = is_train;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (
		net_path, 
		is_train?caffe::TRAIN:caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
}
void CaffeParser::find_inputs(
	const vector<boost::shared_ptr<caffe::Layer<float> >>& layers,
	const std::vector<std::string>& layer_names,
	const std::vector<std::string>& blob_names,
	const std::vector<int>& bottom_ids,
	std::map<std::string, int>& inputs,
	int cur_layer){
	for(int i=0;i<bottom_ids.size();++i){
		const string& bottom_blob_name = blob_names[bottom_ids[i]];
		int inputs_size = inputs.size();
		for(int ii=cur_layer-1;ii>=0;--ii){
			const vector<int>& top_ids = net->top_ids(ii);
			for(int jj = 0;jj<top_ids.size();++jj){
				const string& top_blob_name = blob_names[top_ids[jj]];
				if(top_blob_name != bottom_blob_name) continue;
				if(layers[ii]->type() == string("Input"))
					inputs[layer_names[ii] + "_" + top_blob_name] = inputs.size();
				else inputs[layer_names[ii]] = inputs.size();
				ii = 0;
				break;
			}
		}
		if(inputs.size() == inputs_size){
			inputs[bottom_blob_name] = inputs.size();
		}
	}
}
void CaffeParser::find_attrs(
	const std::string& type,
	const caffe::LayerParameter& caffe_attr,
	std::map<std::string, ParamUnit>& attrs){
	if(find_attrs_funcs.find(type) == find_attrs_funcs.end()){
		printf("no layer: %s in find_attrs_funcs\n",type.c_str());
		exit(0);
	}else if(type != "Input") attrs["name"] = caffe_attr.name();
	find_attrs_func func = find_attrs_funcs[type];
	(this->*func)(caffe_attr,attrs);
}
void CaffeParser::find_params(
	const std::string& type, 
	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& param_blobs,
	std::map<std::string, std::vector<int> >&  params){
	if((caffe_param_table.find(type) == caffe_param_table.end()) && (param_blobs.size() > 0)){
		printf("no layer: %s in caffe_param_table\n",type.c_str());
		exit(0);
	}
	for(int i=0;i<param_blobs.size();++i){
		const caffe::Blob<float> *blob = param_blobs[i].get();
		params[caffe_param_table[type][i]] = {blob->num(),blob->channels(),blob->height(),blob->width()};
	}
}
void CaffeParser::convert(){
	const vector<boost::shared_ptr<caffe::Layer<float> >>& layers = net->layers();
	const vector<string>& layer_names = net->layer_names(), &blob_names = net->blob_names();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();
	parser.set_net_name(net->name());
	{
		LayerUnit u;
		u.attrs = {
			{"type",string("data")},
			{"name", string("data")},
			{"method",string("data")},
			{"n",net->input_blobs()[0]->num()},
			{"c",net->input_blobs()[0]->channels()},
			{"h",net->input_blobs()[0]->height()},
			{"w",net->input_blobs()[0]->width()},
		};
		parser.add_layer(u);
	}
	{
		LayerUnit u;
		u.attrs = {
			{"type",string("data")},
			{"name", string("im_info")},
			{"method",string("im_info")},
			{"n",net->input_blobs()[1]->num()},
			{"c",net->input_blobs()[1]->channels()},
			{"h",net->input_blobs()[1]->height()},
			{"w",net->input_blobs()[1]->width()},
		};
		parser.add_layer(u);
	}

	for(int i=0;i<layers.size();++i){
		string layer_type = layers[i]->type(), layer_name = layer_names[i];
		if(layer_type == "Input") continue;
		LayerUnit u;
		find_attrs(layer_type, layers[i]->layer_param(), u.attrs);
		find_params(layer_type, layers[i]->blobs(), u.params);
		find_inputs(layers, layer_names, blob_names, net->bottom_ids(i),u.inputs,i);
		//for(auto i:attrs)
		//	printf("%s %s %s\n",layer_name.c_str(), i.first.c_str(), i.second.type.c_str());
		//for(auto i:params) printf("%s [%s %d %d %d %d]\n",layer_name.c_str(), i.first.c_str(),i.second[0],i.second[1],i.second[2],i.second[3]);
		//for(auto i:inputs) printf("%s [%s %d]\n",layer_name.c_str(), i.first.c_str(),i.second);
		parser.add_layer(u);
	}
}
void CaffeParser::write_blob(const std::string& layer_name, const std::string& blob_name, const caffe::Blob<float> *blob, FILE* file){
	int total = blob->num() *  blob->channels() * blob->height() *  blob->width();
	char buffer[100];
	sprintf(buffer,"%s %s %d",layer_name.c_str(),blob_name.c_str(),total);
	fwrite(buffer, sizeof(char), 100, file);
	fwrite(blob->cpu_data(), sizeof(float), total, file);
}
void CaffeParser::write_model(const std::string& model_path){
	FILE* file = fopen(model_path.c_str(), "wb");
	const vector<boost::shared_ptr<caffe::Layer<float> >>& layers = net->layers();
	for(int i=0;i<layers.size();++i){
		const string& type = layers[i]->type();
		if((caffe_param_table.find(type) == caffe_param_table.end()) && (layers[i]->blobs().size() > 0)){
			printf("no layer: %s in caffe_param_table\n",type.c_str());
			exit(0);
		}
		for(int j=0;j<layers[i]->blobs().size();++j)
			write_blob(layers[i]->layer_param().name(),caffe_param_table[type][j],layers[i]->blobs()[j].get(),file);
	}
	fclose(file);
}
void CaffeParser::write(const std::string& net_path, const std::string& model_path){
	parser.write_net(net_path);
	write_model(model_path);
}
void CaffeParser::find_conv_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::ConvolutionParameter& p = caffe_attr.convolution_param();
	int kernel_size = p.kernel_size().size() == 1?p.kernel_size()[0]:1,
	    pad = p.pad().size() == 1? p.pad()[0]:0,
	    stride = p.stride().size() == 1? p.stride()[0]:1;
	attrs.insert({
		{"type", string("conv")},
		{"num", p.num_output()},
		{"kernel_size_h", p.has_kernel_h()?p.kernel_h():kernel_size},
		{"kernel_size_w", p.has_kernel_w()?p.kernel_w():kernel_size},
		{"pad_h", p.has_pad_h()?p.pad_h():pad},
		{"pad_w", p.has_pad_w()?p.pad_w():pad},
		{"stride_h", p.has_stride_h()?p.stride_h():stride},
		{"stride_w", p.has_stride_w()?p.stride_w():stride},
		{"bias", p.bias_term()},
		{"group", p.group()}
		});
}
void CaffeParser::find_pooling_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::PoolingParameter& p = caffe_attr.pooling_param();
	string method;
	switch (p.pool()){
		case caffe::PoolingParameter_PoolMethod_MAX:
		method = "max"; break;
		case caffe::PoolingParameter_PoolMethod_AVE:
		method = "avg"; break;
		default:
		printf("cannot process pooling method in find_pooling_attrs\n");
		exit(0);
	}
	attrs.insert({
		{"type",string("pool")},
		{"global",p.global_pooling()},
		{"stride",int(p.stride())},
		{"pad",int(p.pad())},
		{"method",method}
		});
	if(!p.global_pooling()) attrs["kernel_size"] = p.kernel_size();
}
void CaffeParser::find_relu_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::ReLUParameter& p = caffe_attr.relu_param();
	attrs.insert({
		{"type",string("activity")},
		{"method",string("relu")},
		{"neg_slope",p.negative_slope()}
		});
}
void CaffeParser::find_fc_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::InnerProductParameter& p = caffe_attr.inner_product_param();
	attrs.insert({
		{"type",string("fc")},
		{"bias",p.bias_term()},
		{"num",int(p.num_output())}
		});
}
void CaffeParser::find_concat_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs.insert({
		{"type",string("concat")},
		{"method",string("channel")},
		});
}
void CaffeParser::find_split_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs["type"] = string("split");
}
void CaffeParser::find_lrn_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::LRNParameter& p = caffe_attr.lrn_param();
	attrs.insert({
		{"type",string("lrn")},
		{"local_size",int(p.local_size())},
		{"alpha",p.alpha()},
		{"beta",p.beta()}
		});
}
void CaffeParser::find_softmaxloss_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs.insert({
		{"type",string("loss")},
		{"method",string("softmax")}
		});
}
void CaffeParser::find_bn_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs.insert({
		{"type",string("bn")},
		{"eps",caffe_attr.batch_norm_param().eps()}
		});
}
void CaffeParser::find_scale_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs.insert({
		{"type",string("scale")},
		{"bias",caffe_attr.scale_param().bias_term()}
		});
}
void CaffeParser::find_eltwise_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::EltwiseParameter& p = caffe_attr.eltwise_param();
	if(p.operation() != caffe::EltwiseParameter_EltwiseOp_SUM){
		printf("no such op in eltwise\n");
		exit(0);
	}
	float coeff0 = 1.0f, coeff1 = 1.0f;
	if(p.coeff().size()){
		coeff0 = p.coeff()[0];
		coeff1 = p.coeff()[1];
	}
	attrs.insert({
		{"type",string("eltwise")},
		{"method",string("sum")},
		{"coeff0",coeff0},
		{"coeff1",coeff1}
		});
}
void CaffeParser::find_softmax_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	attrs.insert({{"type",string("softmax")}});
}
void CaffeParser::find_python_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::PythonParameter& p = caffe_attr.python_param();
	if(p.layer()!="ProposalLayer"){
		printf("cannot parser python layer != ProposalLayer\n");
		exit(0);
	}
	string feat_stride_param = p.param_str();
	int feat_stride = atoi(feat_stride_param.substr(feat_stride_param.find_last_of(' '),feat_stride_param.size()-feat_stride_param.find_last_of(' ')+1).c_str());
	attrs.insert({
		{"type",string("proposal")},
		{"feat_stride",feat_stride},
		});
}
void CaffeParser::find_reshape_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::ReshapeParameter& p = caffe_attr.reshape_param();
	if(p.shape().dim().size()!=4){
		printf("cannot parser reshape dim != 4\n");
		exit(0);
	}
	attrs.insert({
		{"type",string("reshape")},
		{"d0",p.shape().dim()[0]},
		{"d1",p.shape().dim()[1]},
		{"d2",p.shape().dim()[2]},
		{"d3",p.shape().dim()[3]},
		});
}
void CaffeParser::find_roipooling_attrs(const caffe::LayerParameter& caffe_attr, std::map<std::string, ParamUnit>& attrs){
	const caffe::ROIPoolingParameter& p = caffe_attr.roi_pooling_param();
	attrs.insert({
		{"type",string("roipooling")},
		{"h",p.pooled_h()},
		{"w",p.pooled_w()},
		{"s",p.spatial_scale()}
		});
}
