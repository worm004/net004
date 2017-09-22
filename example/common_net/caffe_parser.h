#include <string>
#include "caffe/caffe.hpp"
#include "Parser.h"
#include "JsonParser.h"
class CaffeParser{
	public:
	CaffeParser();
	void load_caffe_model(
		const std::string& net_path, 
		const std::string& model_path, 
		bool is_train);
	void write(const std::string& net_path, const std::string& model_path);
	void convert();
	void find_inputs(
		const std::vector<boost::shared_ptr<caffe::Layer<float> >>& layers,
		const std::vector<std::string>& layer_names, 
		const std::vector<std::string>& blob_names, 
		const std::vector<int>& bottom_ids, 
		std::map<std::string, int>& inputs,
		int cur_layer);
	void find_inputs(
		const std::vector<boost::shared_ptr<caffe::Layer<float> >>& layers,
		const std::vector<std::string>& layer_names, 
		const std::vector<std::string>& blob_names, 
		const std::vector<int>& bottom_ids, 
		JsonValue& inputs,
		int cur_layer);
	void find_params(
		const std::string& type, 
		const std::vector<boost::shared_ptr<caffe::Blob<float> > >& param_blobs,
		JsonValue& params);
	void find_attrs(
		const std::string& type,
		const caffe::LayerParameter& caffe_attr,
		JsonValue& attrs);
	void write_model(const std::string& model_path);
	void write_blob(const std::string& layer_name, const std::string& blob_name, const caffe::Blob<float> *blob, FILE* file);

#define func_param const caffe::LayerParameter& param, JsonValue& params
	typedef void (CaffeParser::*find_attrs_func) (func_param);
	void find_conv_attrs(func_param);
	void find_pooling_attrs(func_param);
	void find_relu_attrs(func_param);
	void find_fc_attrs(func_param);
	void find_concat_attrs(func_param);
	void find_split_attrs(func_param);
	void find_lrn_attrs(func_param);
	void find_softmaxloss_attrs(func_param);
	void find_bn_attrs(func_param);
	void find_scale_attrs(func_param);
	void find_eltwise_attrs(func_param);
#undef func_param

	private:
	JsonParser jparser;
	std::shared_ptr<caffe::Net<float> > net;
	bool is_train = false;
	std::map<std::string,std::vector<std::string> > caffe_param_table;
	std::map<std::string, find_attrs_func> find_attrs_funcs;

};
