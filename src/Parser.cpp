#include <vector>
#include <set>
#include "Parser.h"
#include "Net004.h"
#include "DataLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "LRNLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"
#include "ConcatLayer.h"
#include "ActivityLayer.h"
#include "my_bigfile_reader.h"

using namespace std;
void Parser::write(Net004* net, const std::string& net_path, const std::string& data_path){	
	Connections &cs = net->cs;
	Layers &ls = net->ls;
	ofstream net_file(net_path);
	ofstream data_file(data_path);
	int n = ls.count();
	for(int i=0;i<n;++i){
		Layer* layer = ls[cs.sorted_cs[i]];
		const string& layer_type = layer->type;
		const string& layer_name = layer->name;
		net_file<<"Layer: "<<layer_type<<" "<<layer_name<<endl;
		if(layer_type == "data"){
			write_net_data(layer, net_file);
			write_dat_data(layer, data_file);
		}
		else if(layer_type == "conv"){
			write_net_conv(layer, net_file);
			write_dat_conv(layer, data_file);
		}
		else if(layer_type == "pool"){
			write_net_pool(layer, net_file);
			write_dat_pool(layer, data_file);
		}
		else if(layer_type == "activity"){
			write_net_activity(layer, net_file);
			write_dat_activity(layer, data_file);
		}
		else if(layer_type == "fc"){
			write_net_fc(layer, net_file);
			write_dat_fc(layer, data_file);
		}
		else if(layer_type == "loss"){
			write_net_loss(layer, net_file);
			write_dat_loss(layer, data_file);
		}
		else{
			printf("no such parser for layer: %s\n",layer_type.c_str());
			exit(0);
		}
	}
	write_connections(net,net_file);
	net_file.close();
	data_file.close();
}
void Parser::write_net_data(Layer* layer, std::ofstream& ofile){
	// if need
	DataLayer *l = (DataLayer*)layer;
	ofile<<l->n<<" "<<l->c<<" "<<l->h<<" "<<l->w<<" "<<l->method<<endl;
}
void Parser::write_net_pool(Layer* layer, std::ofstream& ofile){
	PoolLayer *l = (PoolLayer*)layer;
	string method = l->get_method();
	ofile<<l->get_kernel()<<" "<<l->get_padding()<<" "<<l->get_stride()<<" "<<method<<endl;
}
void Parser::write_net_activity(Layer* layer, std::ofstream& ofile){
	ActivityLayer *l = (ActivityLayer*)layer;
	string method = l->get_method();
	ofile<<method<<endl;
}
void Parser::write_net_fc(Layer* layer, std::ofstream& ofile){
	FCLayer *l = (FCLayer*)layer;
	string activity = l->get_activity();
	if(activity.size()==0) activity = "none";
	ofile<<l->get_n()<<" "<<activity<<endl;
}
void Parser::write_net_loss(Layer* layer, std::ofstream& ofile){
	LossLayer *l = (LossLayer*)layer;
	string method = l->get_method();
	ofile<<method<<endl;
}
void Parser::write_net_conv(Layer* layer, std::ofstream& ofile){
	ConvLayer *l = (ConvLayer*)layer;
	string activity = l->get_activity();
	if(activity.size()==0) activity = "none";
	ofile<<l->get_kernel()<<" "<<l->get_filters()<<" "<<l->get_padding()<<" "<<l->get_stride()<<" "<<activity<<endl;
}
void Parser::write_connections(Net004* net, std::ofstream& ofile){
	Connections &cs = net->cs;
	Layers &ls = net->ls;
	ofile<<"Connections:"<<endl;

	const vector<string> &sorted_cs = cs.sorted_cs;
	int n = sorted_cs.size();
	for(int i=0;i<n;++i){
		const string &from = sorted_cs[i];
		if(!cs.exist(from)) continue;
		const set<string> &tos = cs[from];
		for(auto to: tos) ofile<<from<<" "<<to<<endl;
	}
}

void Parser::write_dat_data(Layer* layer, std::ofstream& ofile){
	// no parameter
}
void Parser::write_dat_conv(Layer* layer, std::ofstream& ofile){
	ConvLayer* l = (ConvLayer*)layer;
	ofile<<"Layer: "<<l->name<<" "<<"weight"<<endl;
	ofile<<l->weight.n<<" "<<l->weight.c<<" "<<l->weight.h<<" "<<l->weight.w<<endl;
	float * wdata = l->weight.data;
	int n = l->weight.nchw();
	for(int i=0;i<n;++i)
		ofile<<wdata[i]<<" ";
	ofile<<endl;

	ofile<<"Layer: "<<l->name<<" "<<"bias"<<endl;
	ofile<<l->bias.n<<" "<<l->bias.c<<" "<<l->bias.h<<" "<<l->bias.w<<endl;
	float * bdata = l->bias.data;
	n = l->bias.nchw();
	for(int i=0;i<n;++i)
		ofile<<bdata[i]<<" ";
	ofile<<endl;
}
void Parser::write_dat_pool(Layer* layer, std::ofstream& ofile){
	// no parameter
}
void Parser::write_dat_activity(Layer* layer, std::ofstream& ofile){
	// no parameter
}
void Parser::write_dat_fc(Layer* layer, std::ofstream& ofile){
	FCLayer* l = (FCLayer*)layer;
	ofile<<"Layer: "<<l->name<<" "<<"weight"<<endl;
	ofile<<l->weight.n<<" "<<l->weight.c<<" "<<l->weight.h<<" "<<l->weight.w<<endl;
	float * wdata = l->weight.data;
	int n = l->weight.nchw();
	for(int i=0;i<n;++i)
		ofile<<wdata[i]<<" ";
	ofile<<endl;

	ofile<<"Layer: "<<l->name<<" "<<"bias"<<endl;
	ofile<<l->bias.n<<" "<<l->bias.c<<" "<<l->bias.h<<" "<<l->bias.w<<endl;
	float * bdata = l->bias.data;
	n = l->bias.nchw();
	for(int i=0;i<n;++i)
		ofile<<bdata[i]<<" ";
	ofile<<endl;
}
void Parser::write_dat_loss(Layer* layer, std::ofstream& ofile){
	// no parameter
}

void Parser::read(const std::string& net_path, const std::string& data_path,Net004* net){
	read_net(net_path,net);
	net->check();
	net->setup();
	//read_data(data_path,net);
	read_data2(data_path,net);
	//net->show();
}
void Parser::read_net(const std::string& path, Net004* net){
	Layers & ls = net->ls;
	Connections& cs = net->cs;

	ifstream net_file(path);
	string line;
	char layer_type[100], layer_name[100];
	bool isconnect = false;
	getline(net_file,line);
	net->name = line;
	while(1){
		getline(net_file,line);
		if(net_file.eof()) break;
		if(line.find("Layer:") == 0) sscanf(line.c_str(),"Layer: %s %s",layer_type, layer_name);
		else if(line.find("Connections:") == 0) isconnect = true;
		else{
			if(isconnect){
				char from[100],to[100];
				sscanf(line.c_str(),"%s %s",from,to);
				cs.add(vector<string>({from,to}));
			}
			else if(layer_type == string("conv")) read_net_conv(line,layer_name,&ls);
			else if(layer_type == string("pool")) read_net_pool(line,layer_name,&ls);
			else if(layer_type == string("activity")) read_net_activity(line,layer_name,&ls);
			else if(layer_type == string("fc")) read_net_fc(line,layer_name,&ls);
			else if(layer_type == string("loss")) read_net_loss(line,layer_name,&ls);
			else if(layer_type == string("data")) read_net_data(line,layer_name,&ls);
			else {
				printf("No such layer to parser %s\n",layer_type);
				exit(0);
			}
		}
	}
	net_file.close();
	cs.update();
}
void Parser::read_data2(const std::string& path, Net004* net){
	Layers & ls = net->ls;
	MY_BIGFILE_READER reader;
	reader.read(path.c_str());
	while(*reader.pbuf){
		char buf[100], layer_name[100], data_name[100];
		reader.to_str(buf,' ');
		reader.to_str(layer_name,' ');
		reader.to_str(data_name,'\n');
		int n = reader.read_int(' ');
		int c = reader.read_int(' ');
		int h = reader.read_int(' ');
		int w = reader.read_int('\n');
		Layer* layer = ls[layer_name];
		Blob* b;
		if(layer->type == "conv"){
			ConvLayer* l = (ConvLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		else if(layer->type == "fc"){
			FCLayer* l = (FCLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		if((b->n != n) || (b->c != c) || (b->h != h) || (b->w != w)){
			printf("layer name: %s\nread: %d %d %d %d\n",layer_name,n,c,h,w);
			b->show();
			printf("Blob size does not match in net and data\n");
			exit(0);
		}
		int total = n*c*h*w;
		for(int i=0;i<total;++i) b->data[i] = reader.read_float(' ');
		reader.to_str(buf,'\n');
	}
}
void Parser::read_data(const std::string& path, Net004* net){
	Layers & ls = net->ls;
	ifstream data_file(path);
	string line;
	while(1){
		getline(data_file,line);
		if(data_file.eof()) break;
		char layer_name[100], data_name[100];
		sscanf(line.c_str(),"Layer: %s %s",layer_name, data_name);

		getline(data_file,line);
		if(data_file.eof()) break;
		int n,c,h,w;
		sscanf(line.c_str(),"%d %d %d %d",&n,&c,&h,&w);

		// checking
		Layer* layer = ls[layer_name];
		Blob* b;
		if(layer->type == "conv"){
			ConvLayer* l = (ConvLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		else if(layer->type == "fc"){
			FCLayer* l = (FCLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		if((b->n != n) || (b->c != c) || (b->h != h) || (b->w != w)){
			printf("layer name: %s\nread: %d %d %d %d\n",layer_name,n,c,h,w);
			b->show();
			printf("Blob size does not match in net and data\n");
			exit(0);
		}
		int total = n*c*h*w;
		for(int i=0;i<total;++i) data_file>>b->data[i];
		getline(data_file, line);
	}
	data_file.close();
}
void Parser::read_net_data(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	int n,c,h,w;
	sscanf(line.c_str(),"%d %d %d %d %s",&n,&c,&h,&w,method);
	ls->add_data(name,n,c,h,w,method);
}
void Parser::read_net_conv(const std::string& line, const std::string& name, Layers* ls){
	char activity[100];
	int kernel, filters, padding, stride;
	sscanf(line.c_str(),"%d %d %d %d %s",&kernel, &filters, &padding, &stride, activity);
	if (activity == string("none")) ls->add_conv(name,{filters,kernel,stride,padding},"");
	else ls->add_conv(name,{filters,kernel,stride,padding},activity);
}
void Parser::read_net_pool(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	int kernel, stride, padding;
	sscanf(line.c_str(),"%d %d %d %s",&kernel, &padding, &stride, method);
	ls->add_pool(name,{kernel, stride, padding},method);
}
void Parser::read_net_activity(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	sscanf(line.c_str(),"%s",method);
	ls->add_activity(name, method);
}
void Parser::read_net_fc(const std::string& line, const std::string& name, Layers* ls){
	char activity[100];
	int n;
	sscanf(line.c_str(),"%d %s",&n, activity);
	if (activity == string("none")) ls->add_fc(name,n,"");
	else ls->add_fc(name,n,activity);
}
void Parser::read_net_loss(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	sscanf(line.c_str(),"%s",method);
	ls->add_loss(name, method);
}
