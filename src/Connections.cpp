#include <iostream>
#include <queue>
#include "stdlib.h"
#include "Connections.h"
using namespace std;
void Connections::clear(){
	cs.clear();
	sorted_cs.clear();
}
Connections& Connections::add(const vector<string>& cs){
	if(cs.size() == 0) return *this;
	else if(cs.size() == 1){
		printf("error: cs.size == 1\n");
		exit(0);
	}
	for(int i = 1;i<cs.size();++i){
		auto& c = this->cs[cs[i-1]];
		c.insert(cs[i]);
	}
	return *this;
}
Connections& Connections::add(const vector<vector<string> >& cs){
	if(cs.size() < 3){
		printf("error: cs.size == 3\n");
		exit(0);
	}
	else if(cs.front().size() != 1){
		printf("error: cs.front.size == 1\n");
		exit(0);
	}
	else if(cs.back().size() != 1){
		printf("error: cs.back.size == 1\n");
		exit(0);
	}
	const string& src = cs.front()[0], des = cs.back()[0];
	auto &c = this->cs[src];

	for(int i=1;i<cs.size()-1;++i){
		if(cs[i].size() > 1) add(cs[i]);
		c.insert(cs[i].front());
		auto &c1 = this->cs[cs[i].back()];
		c1.insert(des);
	}
	return *this;
}
void Connections::update(){
	if(!tsort()){
		printf("error: there is loop in the net\n");
		exit(0);
	}
}
void Connections::show(){
	printf("connections:\n");
	for(int i=0; i < sorted_cs.size();++i){
		const string& last = sorted_cs[i-1];
		const string& cur = sorted_cs[i];
		string pre;
		if(i == 0);
		else if(exist(last,cur))
			pre = "-->";
		else pre = "\n";

		cout<<pre<<"["<<cur<<"]";
	}
	cout<<endl;
}
std::set<std::string> & Connections::operator [](const std::string& name){
	if(cs.find(name) == cs.end()){
		printf("error: connections cannot find layer: %s\n",name.c_str());
		exit(0);
	}
	return cs[name];
}
void Connections::outdegrees(std::map<std::string, int>& outs){
	outs.clear();
	for(const auto& src: cs){
		if(outs.find(src.first) == outs.end()) outs[src.first] = 0;
		for(const auto& to: src.second)
			outs[src.first] += 1;
	}
}
void Connections::indegrees(std::map<std::string, int>& ins){
	ins.clear();
	for(const auto& src: cs){
		if(ins.find(src.first) == ins.end()) ins[src.first] = 0;
		for(const auto& to: src.second){
			if(ins.find(to) == ins.end()) ins[to] = 0;
			ins[to] += 1;
		}
	}
}
bool Connections::tsort(){
	sorted_cs.clear();
	map<string, int> ins;
	indegrees(ins);

	set<string> noins;
	for(const auto& l: ins)
		if(l.second == 0) noins.insert(l.first);
	while(!noins.empty()){
		const string l = *noins.begin();
		sorted_cs.push_back(l);
		noins.erase(noins.begin());
		if(cs.find(l) == cs.end()) continue;
		for(const auto& to : cs[l]){
			ins[to] -= 1;
			if(ins[to] == 0) {
				noins.insert(to);
			}
		}
	}

	bool ret = true;
	for(const auto&l:ins){
		if(l.second !=0) {
			ret = false;
		}
	}
	return ret;

}
bool Connections::exist(const std::string& src){
	return cs.find(src) != cs.end();
}
bool Connections::exist(const string& src, const string& des){
	if(cs.find(src) == cs.end()) return false;
	auto & c = cs[src];
	return c.find(des) != c.end();
}
