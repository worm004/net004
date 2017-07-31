#ifndef CONNECTIONS_H
#define CONNECTIONS_H
#include <string>
#include <vector>
#include <map>
#include <set>
class Connections{
	public:
	Connections& add(const std::vector<std::string>& cs);
	Connections& add(const std::vector<std::vector<std::string> >& cs);
	void update();
	void clear();
	void show();
	void indegrees(std::map<std::string, int>& ins);
	void outdegrees(std::map<std::string, int>& outs);
	std::set<std::string> & operator [](const std::string& name);
	bool exist(const std::string& src);

	private:
	bool exist(const std::string& src, const std::string& des);
	bool tsort();

	public:
	std::vector<std::string> sorted_cs;

	private:
	std::map<std::string, std::set<std::string> > cs;
};
#endif
