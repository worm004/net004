#include "stdlib.h"
#include <map>
#include "caffe_parser.h"
using namespace std;
CaffeParser::CaffeParser(){
	caffe_param_table = {
		{"Convolution", {"weight","bias"}},
		{"Deconvolution", {"weight","bias"}},
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
		{"Softmax", &CaffeParser::find_softmax_attrs},
		{"Crop", &CaffeParser::find_crop_attrs},
		{"Deconvolution", &CaffeParser::find_dconv_attrs}
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
	const std::vector<boost::shared_ptr<caffe::Layer<float> >>& layers,
	const std::vector<std::string>& layer_names, 
	const std::vector<std::string>& blob_names, 
	const std::vector<int>& bottom_ids, 
	JsonValue& inputs,
	int cur_layer){
	for(int i=0;i<bottom_ids.size();++i){
		const string& bottom_blob_name = blob_names[bottom_ids[i]];
		for(int ii=cur_layer-1;ii>=0;--ii){
			const vector<int>& top_ids = net->top_ids(ii);
			for(int jj = 0;jj<top_ids.size();++jj){
				const string& top_blob_name = blob_names[top_ids[jj]];
				if(top_blob_name != bottom_blob_name) continue;
				int size = inputs.jobj.size();
				if(layers[ii]->type() == string("Input"))
					inputs.jobj[layer_names[ii] + "_" + top_blob_name] = JsonValue("v",size);
				else inputs.jobj[layer_names[ii]] = JsonValue("v",size);
				ii = 0;
				break;
			}
		}
	}
}
void CaffeParser::find_params(
	const std::string& type, 
	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& param_blobs,
	JsonValue& params){
	if((caffe_param_table.find(type) == caffe_param_table.end()) && (param_blobs.size() > 0)){
		printf("no layer: %s in caffe_param_table\n",type.c_str());
		exit(0);
	}
	for(int i=0;i<param_blobs.size();++i){
		const caffe::Blob<float> *blob = param_blobs[i].get();
		JsonValue array("array");
		array.jarray = {
				{"v",(double)blob->num()},
				{"v",(double)blob->channels()},
				{"v",(double)blob->height()},
				{"v",(double)blob->width()}
				};
		params.jobj[caffe_param_table[type][i]] = array;
	}
}
void CaffeParser::find_attrs(
	const std::string& type,
	const caffe::LayerParameter& caffe_attr,
	JsonValue& attrs){
	if(find_attrs_funcs.find(type) == find_attrs_funcs.end()){
		printf("no layer: %s in find_attrs_funcs\n",type.c_str());
		exit(0);
	}else if(type != "Input") attrs.jobj["name"] = JsonValue("v",caffe_attr.name());
	find_attrs_func func = find_attrs_funcs[type];
	(this->*func)(caffe_attr,attrs);
}
void CaffeParser::convert(){
	const vector<boost::shared_ptr<caffe::Layer<float> >>& layers = net->layers();
	const vector<string>& layer_names = net->layer_names(), &blob_names = net->blob_names();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();
	jparser.j = JsonValue("obj");
	jparser.j.jobj["net_name"] = {"v",net->name()};
	jparser.j.jobj["layers"] = JsonValue("array");
	for(int i=0;i<layers.size();++i){
		if(layers[i]->type() != string("Input")) continue;
		string layer_name = layer_names[i];
		for(int j=0;j<layers[i]->layer_param().top().size();++j){
			const string& top_name = layers[i]->layer_param().top()[j];
			string method = top_name.find("label") != string::npos? "label":"data";
			JsonValue jo("obj");
			jo.jobj["attrs"] = JsonValue("obj");
			jo.jobj["attrs"].jobj = {
				{"type",{"v","data"}},
				{"name",{"v",layer_name + "_" + top_name}},
				{"method",{"v",method}},
				{"n",{"v",double(tops[i][j]->num())}},
				{"c",{"v",double(tops[i][j]->channels())}},
				{"h",{"v",double(tops[i][j]->height())}},
				{"w",{"v",double(tops[i][j]->width())}}
				};
			jparser.j.jobj["layers"].jarray.push_back(jo);
		}
	}
	for(int i=0;i<layers.size();++i){
		string layer_type = layers[i]->type(), layer_name = layer_names[i];
		if(layer_type == "Input") continue;
		JsonValue j("obj");
		j.jobj["attrs"] = JsonValue("obj");
		j.jobj["params"] = JsonValue("obj");
		j.jobj["inputs"] = JsonValue("obj");
		JsonValue &jattrs = j.jobj["attrs"], &jparams = j.jobj["params"], &jinputs = j.jobj["inputs"];

		find_attrs(layer_type, layers[i]->layer_param(), jattrs);
		find_params(layer_type, layers[i]->blobs(), jparams);
		find_inputs(layers, layer_names, blob_names, net->bottom_ids(i),jinputs,i);
		jparser.j.jobj["layers"].jarray.push_back(j);
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
	jparser.write(net_path);
	write_model(model_path);
}
void CaffeParser::find_conv_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::ConvolutionParameter& p = caffe_attr.convolution_param();
	int kernel_size = p.kernel_size().size() == 1?p.kernel_size()[0]:1,
	    pad = p.pad().size() == 1? p.pad()[0]:0,
	    stride = p.stride().size() == 1? p.stride()[0]:1;
	attrs.jobj.insert({
		{"type", {"v","conv"}},
		{"num", {"v",double(p.num_output())}},
		{"kernel_size_h", {"v",double(p.has_kernel_h()?p.kernel_h():kernel_size)}},
		{"kernel_size_w", {"v",double(p.has_kernel_w()?p.kernel_w():kernel_size)}},
		{"pad_h", {"v",double(p.has_pad_h()?p.pad_h():pad)}},
		{"pad_w", {"v",double(p.has_pad_w()?p.pad_w():pad)}},
		{"stride_h", {"v",double(p.has_stride_h()?p.stride_h():stride)}},
		{"stride_w", {"v",double(p.has_stride_w()?p.stride_w():stride)}},
		{"bias", {"v",double(p.bias_term())}},
		{"group", {"v",double(p.group())}}
		});
}
void CaffeParser::find_pooling_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
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
	attrs.jobj.insert({
		{"type",{"v","pool"}},
		{"global",{"v",double(p.global_pooling())}},
		{"stride",{"v",double(p.stride())}},
		{"pad",{"v",double(p.pad())}},
		{"method",{"v",method}}
		});
	if(!p.global_pooling()) attrs.jobj["kernel_size"] = {"v",double(p.kernel_size())};
}
void CaffeParser::find_relu_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::ReLUParameter& p = caffe_attr.relu_param();
	attrs.jobj.insert({
		{"type",{"v","activity"}},
		{"method",{"v","relu"}},
		{"neg_slope",{"v",p.negative_slope()}}
		});
}
void CaffeParser::find_fc_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::InnerProductParameter& p = caffe_attr.inner_product_param();
	attrs.jobj.insert({
		{"type",{"v","fc"}},
		{"bias",{"v",double(p.bias_term())}},
		{"num",{"v",double(p.num_output())}}
		});
}
void CaffeParser::find_concat_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj.insert({
		{"type",{"v","concat"}},
		{"method",{"v","channel"}},
		});
}
void CaffeParser::find_split_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj["type"] = {"v","split"};
}
void CaffeParser::find_lrn_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::LRNParameter& p = caffe_attr.lrn_param();
	attrs.jobj.insert({
		{"type",{"v","lrn"}},
		{"local_size",{"v",double(p.local_size())}},
		{"alpha",{"v",p.alpha()}},
		{"beta",{"v",p.beta()}}
		});
}
void CaffeParser::find_softmaxloss_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj.insert({
		{"type",{"v","loss"}},
		{"method",{"v","softmax"}}
		});
}
void CaffeParser::find_bn_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj.insert({
		{"type",{"v","bn"}},
		{"eps",{"v",caffe_attr.batch_norm_param().eps()}}
		});
}
void CaffeParser::find_scale_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj.insert({
		{"type",{"v","scale"}},
		{"bias",{"v",(double)caffe_attr.scale_param().bias_term()}}
		});
}
void CaffeParser::find_eltwise_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
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
	attrs.jobj.insert({
		{"type",{"v","eltwise"}},
		{"method",{"v","sum"}},
		{"coeff0",{"v",(double)coeff0}},
		{"coeff1",{"v",(double)coeff1}}
		});
}
void CaffeParser::find_softmax_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	attrs.jobj.insert({{"type",{"v","softmax"}}});
}
void CaffeParser::find_crop_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::CropParameter& p = caffe_attr.crop_param();
	if(p.offset().size() != 1){
		printf("cannot process offset > 1 in crop\n");
		exit(0);
	}
	attrs.jobj.insert({
		{"type",{"v","crop"}},
		{"axis",{"v",double(p.axis())}},
		{"offset",{"v",double(p.offset()[0])}},
		});
}
void CaffeParser::find_dconv_attrs(const caffe::LayerParameter& caffe_attr, JsonValue& attrs){
	const caffe::ConvolutionParameter& p = caffe_attr.convolution_param();
	int kernel_size = p.kernel_size().size() == 1?p.kernel_size()[0]:1,
	    pad = p.pad().size() == 1? p.pad()[0]:0,
	    stride = p.stride().size() == 1? p.stride()[0]:1;
	attrs.jobj.insert({
		{"type", {"v","dconv"}},
		{"num", {"v",double(p.num_output())}},
		{"kernel_size_h", {"v",double(p.has_kernel_h()?p.kernel_h():kernel_size)}},
		{"kernel_size_w", {"v",double(p.has_kernel_w()?p.kernel_w():kernel_size)}},
		{"pad_h", {"v",double(p.has_pad_h()?p.pad_h():pad)}},
		{"pad_w", {"v",double(p.has_pad_w()?p.pad_w():pad)}},
		{"stride_h", {"v",double(p.has_stride_h()?p.stride_h():stride)}},
		{"stride_w", {"v",double(p.has_stride_w()?p.stride_w():stride)}},
		{"bias", {"v",double(p.bias_term())}},
		{"group", {"v",double(p.group())}}
		});
}
