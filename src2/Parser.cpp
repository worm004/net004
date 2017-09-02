#include <fstream>
#include <sstream>
#include "Parser.h"
#include "Net004.h"
#include <stack>
using namespace std;

void NetParser::add_layer(const LayerUnit& u){
	layers.push_back(u);
}

void NetParser::write_net(const std::string& path){
#define offset(t) for(int i=0;i<(t);++i) ss<<"  "
	ofstream ofile(path);
	int t = 0;
	stringstream ss;
	ss << "{"<<endl; ++t;
	offset(t); ss<<"\"net_name\":\""<<net_name<<"\","<<endl;
	offset(t); ss<<"\"layers\":["<<endl; ++t;
	for(int i=0;i<layers.size();++i){
		offset(t); ss<<"{"<<endl; ++t;
		// attrs
		offset(t); ss<<"\"attrs\":{"<<endl;++t;
		if(layers[i].attrs.find("type")==layers[i].attrs.end()){
			printf("layers should have type\n");
			exit(0);
		}
		else if(layers[i].attrs.find("name")==layers[i].attrs.end()){
			printf("layers should have name\n");
			exit(0);
		}
		offset(t); ss<<"\"type\":\""<<layers[i].attrs["type"].sval<<"\","<<endl;
		offset(t); ss<<"\"name\":\""<<layers[i].attrs["name"].sval<<((layers[i].attrs.size() == 2)?"\"":"\",")<<endl;
		for(const auto& j: layers[i].attrs){
			if(j.first == "type" || j.first == "name") continue;
			offset(t);
			ss<<"\""<<j.first<<"\":";
			if(j.second.type == "float") ss<<j.second.fval;
			else if(j.second.type == "string") ss<<"\""<<j.second.sval<<"\"";
			else{
				printf("unknown value type: %s\n",j.second.type.c_str());
				exit(0);
			}
			ss<<((&j == &*(prev(layers[i].attrs.end())))?"":",")<<endl;
		}
		--t; offset(t); ss<<((layers[i].params.size() + layers[i].inputs.size() > 0)?"},":"}")<<endl;

		// params
		if(layers[i].params.size()){
			offset(t); ss<<"\"params\":{"<<endl;++t;
			for(const auto& j:layers[i].params){
				offset(t); 
				ss<<"\""<<j.first<<"\":["<<
					j.second[0]<<","<<
					j.second[1]<<","<<
					j.second[2]<<","<<
					j.second[3]<<((&j==&*prev(layers[i].params.end()))?"]":"],")<<endl;

			}
			--t; offset(t); ss<<((layers[i].inputs.size() > 0)?"},":"}")<<endl;
		}

		// inputs
		if(layers[i].inputs.size()){
			offset(t); ss<<"\"inputs\":{"<<endl;++t;
			for(const auto& j:layers[i].inputs){
				offset(t); 
				ss<<"\""<<j.first<<"\":"<< j.second<<((&j == &*(prev(layers[i].inputs.end())))?"":",")<<endl;
			}
			--t; offset(t); ss<<"}"<<endl;
		}

		--t; offset(t);ss<<((i==layers.size()-1)?"}":"},")<<endl;
	}

	--t; offset(t); ss<<"]"<<endl;
	--t; ss << "}"<<endl;
	ofile <<ss.rdbuf();
#undef offset
}
void NetParser::read_net(const std::string& path){
#define format_err {printf("format error when reading (%s : %d)\n",__func__,__LINE__);exit(0);}
	map<char,char> opps = { {'{','}'}, {'[',']'}, {'}','{'}, {']','['} };
	stack<char> levels;
	LayerUnit u;
	ifstream file(path);
	int state = -1;
	while(1){
		string raw_line;
		getline(file,raw_line);
		if(file.eof())break;
		int b = 0, e = raw_line.size() - 1;
		while((b<=e) && ((raw_line[b] == ' ') || (raw_line[b] == '\t'))) ++b;
		while((b<=e) && ((raw_line[e] == ' ') || (raw_line[e] == '\t') || (raw_line[e] == ','))) --e;
		if(e<b) continue;
		string line = raw_line.substr(b,e-b+1);
		if(line[0] == '#')continue;
		if(line.find(':') == string::npos){
			for(int i=0;i<line.size();++i){
				char c0 = line[i];
				if((c0 == '{') || (c0 == '[')) levels.push(c0);
				else if((c0 == '}') || (c0 == ']')){
					if(levels.top() != opps[c0]) format_err;
					levels.pop();
					if(levels.size() == 2){
						layers.push_back(u);
						u.clear();
					}
				}
			}
		}
		else{
			int pos = line.find_first_of(":");
			if(line[0]!='"') format_err;
			int b0 = 1, e0 = pos-1;
			while((b0<=e0)&& (line[e0] != '"')) --e0;
			--e0;
			string key = line.substr(b0,e0-b0+1);
			//printf("level: %d, key: %s\n",levels.size(), key.c_str());
			if(levels.size() == 3){
				if(key == "attrs") state = 0;
				else if(key == "inputs") state = 1;
				else if(key == "params") state = 2;
			}else if(levels.size() == 1){
				if(key == "net_name") state = 3;
			}
			string val_part = line.substr(pos+1,line.size()-(pos+1)+1);
			if(val_part[0] == '"'){
				int b1 = 0, e1 = val_part.size()-1;
				while((b1<=e1) && (val_part[b1] != '"')) ++b1;
				while((b1<=e1) && (val_part[e1] != '"')) --e1;
				++b1; --e1;
				if(e1<b1){
					printf("format error when reading (type C)\n");
					exit(0);
				}
				string val = val_part.substr(b1,e1-b1+1);
				if(state == 3) net_name = val;
				else if(state == 0)u.attrs[key] = val;
				else format_err;
			}
			else if(val_part[0] == '['){
				levels.push(val_part[0]);
				if(val_part.size() != 1){
					int n,h,c,w;
					sscanf(val_part.c_str(),"[%d,%d,%d,%d]",&n,&c,&h,&w);
					//printf("val: %d %d %d %d\n",n,c,h,w);
					if(state == 2) u.params[key] = {n,c,h,w};
					else format_err;
					if(val_part[val_part.size()-1] != ']') format_err;
					levels.pop();
				}
			}
			else if(val_part[0] == '{'){
				levels.push(val_part[0]);
				if(val_part.size() != 1) format_err;
			}
			else{
				string val = val_part;
				if(state == 1) u.inputs[key] = atoi(val.c_str());
				else if(state == 0) u.attrs[key] = atof(val.c_str());
				else format_err;
				//printf("val: %s\n",val.c_str());
			}
		}
		//printf("---%s\n",line.c_str());
		//printf("---%d %c\n",levels.size(),levels.top());
	}
	file.close();
#undef format_err
}
void NetParser::set_net_name(const std::string& name){
	net_name = name;
}
void NetParser::set_net_mode(bool is_train){
	this->is_train = is_train;
}
std::string NetParser::get_net_name(){
	return net_name;
}
bool NetParser::get_net_mode(){
	return is_train;
}
const std::vector<LayerUnit>& NetParser::get_layers(){
	return layers;
}
ModelParser::ModelParser(){
}
void ModelParser::read_model(const std::string& path, Net004* net){
	FILE* file = fopen(path.c_str(), "rb");
	while(!feof(file)){
		char buffer[100],layer_name[100],data_name[100];
		int n;
		fread(buffer, sizeof(char), 100, file);
		sscanf(buffer,"%s %s %d",layer_name,data_name,&n);
		//printf("%s %s %d\n",layer_name,data_name,n);
		Blob& b = net->operator[](layer_name)->params[data_name];
		if(n!=b.nchw()){
			printf("blob size different %d (%d * %d * %d * %d -> %d)\n",n,b.n,b.c,b.h,b.w,b.nchw());
			exit(0);
		}
		fread(b.data,sizeof(float),n,file);
	}
	fclose(file);
}
