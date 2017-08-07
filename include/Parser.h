#include <map>
#include <fstream>
#include <string>

class Net004;
class Layer;
class Layers;
class Parser{
	typedef void (Parser::*read_net_func) (const std::string&line, const std::string& name, Layers* ls);

	public:
	Parser();
	void write(Net004* net, const std::string& net_path, const std::string& data_path);
	void read(const std::string& net_path, const std::string& data_path,Net004* net);
	void clear();
	int batch_size = -1;

	private:
	void write_connections(Net004* net, std::ofstream& file);
#define params Layer* layer, std::ofstream& ofile
	void write_net_data(params);
	void write_net_conv(params);
	void write_net_pool(params);
	void write_net_activity(params);
	void write_net_fc(params);
	void write_net_loss(params);
#undef params 
#define params Layer* layer, FILE* ofile
	void write_dat_data(params);
	void write_dat_conv(params);
	void write_dat_pool(params);
	void write_dat_activity(params);
	void write_dat_fc(params);
	void write_dat_loss(params);
#undef params 

	void read_net(const std::string& path, Net004* net);
	void read_data(const std::string& path, Net004* net);
#define params const std::string& line, const std::string& name, Layers* ls
	void read_net_data(params);
	void read_net_conv(params);
	void read_net_pool(params);
	void read_net_activity(params);
	void read_net_fc(params);
	void read_net_loss(params);
	void read_net_lrn(params);
	void read_net_split(params);
	void read_net_concat(params);
#undef params 

	private:
	std::map<std::string, read_net_func> read_net_funcs;
};
