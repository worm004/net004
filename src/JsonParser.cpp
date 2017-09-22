#include "stdlib.h"
#include <fstream>
#include <unordered_set>
#include <stack>
#include "JsonParser.h"
using namespace std;
JsonValue::JsonValue(){
}
JsonValue::JsonValue(const std::string& type){
	if((type == "null") || (type == "v") || (type == "array") || (type == "obj"))
		this->type = type;
	else {
		printf("not valid type\n");
		exit(0);
	}
}
JsonValue::JsonValue(const std::string& type, double d){
	if(type != "v"){
		printf("only v uses this constructor\n");
		exit(0);
	}
	this->type = type;
	jv.type = "num";
	jv.d = d;
}
JsonValue::JsonValue(const std::string& type, const std::string& s){
	if(type != "v"){
		printf("only v uses this constructor\n");
		exit(0);
	}
	this->type = type;
	jv.type = "string";
	jv.s = s;
}
std::string JsonPrimitiveValue::to_str(){
	if(type == "string") return "\""+s+"\"";
	else if(type == "num") return to_string(d);
	else{
		printf("error: should not reach here\n");
		exit(0);
	}
}
void find_str(const char*b, const char*e,int &ib, int&ie){
	const char*c = b;
	while((c <= e) && (*c == ' ')) ++c;
	if(*c != '"'){
		printf("error1: there are chars before \"string\"\n");
		exit(0);
	}
	ib = c - b;
	++c;
	while(c<=e){
		if(*c == '\\'){
			if(c + 1>= e){
				printf("error2: cannot find end of string\n");
				exit(0);
			}
			++c;
			if(*c == 'u'){
				if(c+3 >= e){
					printf("error2: cannot find end of string\n");
					exit(0);
				}
				c+=3;
			}
		}
		else if(*c == '"'){
			ie = c - b;
			break;
		}
		++c;
	}
}
std::string JsonValue::check_type(const char*b, const char*e,int &ib, int &ie,bool has_colon){
	const char*c = b;
	while((c<=e)&&(*c == ' ')) ++c;
	if(has_colon){
		if (*c!=':') {
			printf("error1: there are chars before : \n");
			exit(0);
		}
		++c;
		while((c<=e)&&(*c == ' ')) ++c;
	}
	ib = c-b;
	if(*c == '[') return "array";
	else if(*c == '{') return "obj";
	else if(*c == '"') {
		int ib2, ie2;
		find_str(c,e,ib2,ie2);
		ie = ie2 + ib;
		return "v";
	}
	else if((*c == '-') || ((47 < *c) && (*c < 58))){
		while((c<=e) && (*c!='}') && (*c != ']') && (*c != ',')) ++c;
		ie = c - 1 - b;
		return "v";
	}
	if(c + 3>e){
		printf("error3: not valid value\n");
		exit(0);
	}
	if((*c != 'n') || (*(c+1) != 'u') || (*(c+2) != 'l')|| (*(c+2) != 'l')){
		printf("error3: not valid value\n");
		exit(0);
	}
	ie = c - b + 3;
	return "null";
}
void JsonValue::set_val(const std::string& v){
	type = "v";
	if(v[0] == '"') {
		jv.type = "string";
		jv.s = v.substr(1,v.size()-2);
	}
	else{
		jv.type = "num";
		jv.d = atof(v.c_str());
	}
}
void JsonValue::set_null(){
	type = "null";
}
void JsonValue::set_by_type(const std::string& type, JsonValue& obj, const char*& c, const char*e, int val_b, int val_e, std::queue<int>& helper){
	if(type == "v") {
		obj.set_val(string(c+val_b,c+val_e+1));
		c += val_e+1;
	}
	else if(type == "null") {
		obj.set_null();
		c += val_e+1;
	}
	else if(type == "array"){
		int len = helper.front()-1;
		obj.set_array(c+1+val_b,len,helper);
		c += val_b + len + 1;
	}
	else if(type == "obj"){
		int len = helper.front()-1;
		obj.set_obj(c+1+val_b,len,helper);
		c += val_b + len + 1;
	}
	else{
		printf("error: should not touch here\n");
		exit(0);
	}
	while((c<=e) && ((*c == ' ') || (*c == ',') || (*c == ']') || (*c == '}')))++c;
}
void JsonValue::set_array(const char* b, int n, std::queue<int>& helper){
	type = "array";
	//if(n == helper.front()-1){
	//	helper.pop();
	//	return;
	//}
	helper.pop();
	const char*c = b,*e = b+n-1;
	while((c <= e) && (*c == ' ')) ++c;
	if(*c == ']') return;
	while(c <= e){
		int val_b, val_e;
		string ctype = check_type(c,e,val_b, val_e,false);
		jarray.resize(jarray.size()+1);
		set_by_type(ctype, jarray.back(), c,e,val_b,val_e,helper);
	}
}
void JsonValue::set_obj(const char*b, int n, std::queue<int>& helper){
	type = "obj";
	//if(n == helper.front()-1){
	//	helper.pop();
	//	return;
	//}
	helper.pop();
	const char*c = b,*e = b+n-1;
	while((c <= e) && (*c == ' ')) ++c;
	if(*c == '}') return;

	while(c <= e){
		int key_b, key_e;
		find_str(c,e,key_b,key_e);
		string ckey = string(c+key_b+1,c+key_e);
		c = c + key_e + 1;
		int val_b, val_e;
		string ctype = check_type(c,e,val_b, val_e,true);
		set_by_type(ctype, jobj[ckey], c,e,val_b,val_e,helper);
	}
}
std::string JsonValue::to_str(int level){
	if(type == "null") return "null";
	else if(type == "v") return jv.to_str();
	else if(type == "obj"){
		string ret;
		ret += "{\n";
		for(auto it = jobj.begin(); it!= jobj.end();++it){
			if(it!=jobj.begin()) ret += ",\n";
			for(int i=0;i<level;++i) ret += "  ";
			ret += "\"" + it->first + "\":" + it->second.to_str(level+1);
		}
		ret += "\n";
		for(int i=1;i<level;++i) ret += "  ";
		ret += "}";
		return ret;
	}
	else if(type == "array"){
		string ret;
		ret += "[\n";
		for(int i=0;i<jarray.size();++i){
			if(i != 0) ret += ",\n";
			for(int j=0;j<level;++j) ret += "  ";
			ret += jarray[i].to_str(level + 1);
		}
		ret += "\n";
		for(int i=1;i<level;++i) ret += "  ";
		ret += "]";
		return ret;
	}
	else{
		printf("error: should not reach here\n");
		exit(0);
	}
}
void JsonParser::read(const std::string& path){
	ifstream ifile(path);
	string s0((std::istreambuf_iterator<char>(ifile)), (std::istreambuf_iterator<char>()));
	int b = 0, e = s0.size()-1;
	while((b <= e) && ((s0[b] != '{') && (s0[b] != '['))) ++b;
	while((b <= e) && ((s0[e] != '}') && (s0[e] != ']'))) --e;

	string s1;
	for(int i=b;i<=e;++i) if((31 < s0[i]) && (s0[i] < 127)) s1 += s0[i];

	gen_helper(s1.c_str(),s1.c_str()+s1.size()-1);
	//{"a":1}
	//0123456
	//b     e
	if(e - b < 6) return;
	const char* p = s1.c_str()+1;
	int len = s1.size()-2;

	if(s1[0] == '{') j.set_obj(p,len,helper);
	else if(s1[0] == '[') j.set_array(p,len,helper);
	else return;
}
void JsonParser::gen_helper(const char*b, const char*e){
	stack<char> cs;
	stack<int> is;
	vector<pair<int,int> > ret;
	const char* c = b;
	while(c<=e){
		if(*c == '"'){
			int ib,ie;
			find_str(c,e,ib,ie);
			c = c + ie;
		}
		if((*c == '[')||(*c == '{')){
			cs.push(*c);
			is.push(c-b);
		}
		else if((*c == ']')||(*c == '}')){
			char left = cs.top();
			if(*c != left+2){
				printf("error: [ { and } ] not matched\n");
				exit(0);
			}
			ret.push_back({int(c-b)-is.top(),is.top()});
			cs.pop();
			is.pop();
		}
		++c;
	}
	std::sort(ret.begin(), ret.end(), [](pair<int,int> &i, pair<int,int> &j) {
		    return i.second < j.second;
	});
	for(auto i:ret) helper.push(i.first);
}
void JsonParser::show(){
	printf("%s\n",j.to_str(1).c_str());
}
void JsonParser::write(const std::string& path){
	ofstream file(path);
	file << j.to_str(1)<<endl;
}
