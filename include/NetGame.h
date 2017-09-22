#ifndef NETGAME_H
#define NETGAME_H
#include <vector>
#include <string>
#include <map>
#include "Net004.h"
#include "BaseRun.h"
class NetGame{
	public:
	NetGame();
	void load(const std::string& path);
	virtual void init() = 0;
	virtual void run();
	
	public:
	std::string name, net_path, type;
	std::vector<std::string> runlist;
	std::map<std::string,Run*> runs;
	int batch_size, max_iter;
	Net004 net;

	private:
	typedef std::map<std::string, Run*(*)(const JsonValue&)> RunTypeMap;
	RunTypeMap run_type_map;
};
class NetTrain:public NetGame{
	public:
	virtual void init();
};
#endif
