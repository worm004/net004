#include <fstream>
#include <string>

class Net004;
class Layer;
class Layers;
class Parser{
	public:
	void write(Net004* net, const std::string& net_path, const std::string& data_path);
	void read(const std::string& net_path, const std::string& data_path,Net004* net);
	void clear();
	int batch_size = -1;

	private:
	void write_connections(Net004* net, std::ofstream& file);
	void write_net_data(Layer* layer, std::ofstream& ofile);
	void write_net_conv(Layer* layer, std::ofstream& ofile);
	void write_net_pool(Layer* layer, std::ofstream& ofile);
	void write_net_activity(Layer* layer, std::ofstream& ofile);
	void write_net_fc(Layer* layer, std::ofstream& ofile);
	void write_net_loss(Layer* layer, std::ofstream& ofile);


	void write_dat_data(Layer* layer, FILE* ofile);
	void write_dat_conv(Layer* layer, FILE* ofile);
	void write_dat_pool(Layer* layer, FILE* ofile);
	void write_dat_activity(Layer* layer, FILE* ofile);
	void write_dat_fc(Layer* layer, FILE* ofile);
	void write_dat_loss(Layer* layer, FILE* ofile);


	void read_net(const std::string& path, Net004* net);
	void read_data(const std::string& path, Net004* net);
	void read_net_data(const std::string& line, const std::string& name, Layers* ls);
	void read_net_conv(const std::string& line, const std::string& name, Layers* ls);
	void read_net_pool(const std::string& line, const std::string& name, Layers* ls);
	void read_net_activity(const std::string& line, const std::string& name, Layers* ls);
	void read_net_fc(const std::string& line, const std::string& name, Layers* ls);
	void read_net_loss(const std::string& line, const std::string& name, Layers* ls);
	void read_net_lrn(const std::string& line, const std::string& name, Layers* ls);
};
