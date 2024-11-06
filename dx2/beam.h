#ifndef DX2_MODEL_BEAM_H
#define DX2_MODEL_BEAM_H
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using Eigen::Vector3d;
using json = nlohmann::json;

class Beam {
// A monochromatic beam
public:
    Beam()=default;
    Beam(double wavelength);
    Beam(Vector3d s0);
    Beam(double wavelength, Vector3d direction, double divergence,
         double sigma_divergence, Vector3d polarization_normal,
         double polarization_fraction, double flux,
         double transmission, double sample_to_source_distance);
    Beam(json beam_data);
    json to_json() const;
    double get_wavelength() const;
    void set_wavelength(double wavelength);
    Vector3d get_s0() const;
    void set_s0(Vector3d s0);

protected:
    double wavelength_{0.0};
    Vector3d sample_to_source_direction_{0.0,0.0,1.0}; //called direction_ in dxtbx
    double divergence_{0.0}; // "beam divergence - be more specific with name?"
    double sigma_divergence_{0.0}; // standard deviation of the beam divergence
    Vector3d polarization_normal_{0.0,1.0,0.0};
    double polarization_fraction_{0.999};
    double flux_{0.0};
    double transmission_{1.0};
    double sample_to_source_distance_{0.0}; // FIXME is this really needed?
};

Beam::Beam(double wavelength) : wavelength_{wavelength} {}

Beam::Beam(Vector3d s0){
    double len = s0.norm();
    wavelength_ = 1.0 / len;
    sample_to_source_direction_ = -1.0 * s0 / len;
}

// full constructor
Beam::Beam(double wavelength, Vector3d direction, double divergence,
    double sigma_divergence, Vector3d polarization_normal,
    double polarization_fraction, double flux,
    double transmission, double sample_to_source_distance)
    : wavelength_{wavelength},
      sample_to_source_direction_{direction},
      divergence_{divergence},
      sigma_divergence_{sigma_divergence},
      polarization_normal_{polarization_normal},
      polarization_fraction_{polarization_fraction},
      flux_{flux}, transmission_{transmission},
      sample_to_source_distance_{sample_to_source_distance} {}

// constructor from json data
Beam::Beam(json beam_data) {
    // minimal required keys
    std::vector<std::string> required_keys = {
        "wavelength"
    };
    for (const auto &key : required_keys) {
        if (beam_data.find(key) == beam_data.end()) {
            throw std::invalid_argument(
                "Key " + key + " is missing from the input beam JSON");
        }
    }
    wavelength_ = beam_data["wavelength"];
    /* additional potential keys
    "direction", "polarization_normal", "polarization_fraction",
    "divergence", "sigma_divergence", "flux", "transmission",
    "sample_to_source_distance"
    */
    if (beam_data.find("direction") != beam_data.end()){
        Vector3d direction{{beam_data["direction"][0], beam_data["direction"][1], beam_data["direction"][2]}};
        sample_to_source_direction_ = direction;
    }
    if (beam_data.find("divergence") != beam_data.end()){
        divergence_ = beam_data["divergence"];
    }
    if (beam_data.find("sigma_divergence") != beam_data.end()){
        sigma_divergence_ = beam_data["sigma_divergence"];
    }
    if (beam_data.find("polarization_normal") != beam_data.end()){
        Vector3d pn{
            {beam_data["polarization_normal"][0],
            beam_data["polarization_normal"][1],
            beam_data["polarization_normal"][2]}};
        polarization_normal_ = pn;
    }
    if (beam_data.find("polarization_fraction") != beam_data.end()){
        polarization_fraction_ = beam_data["polarization_fraction"];
    }
    if (beam_data.find("flux") != beam_data.end()){
        flux_ = beam_data["flux"];
    }
    if (beam_data.find("transmission") != beam_data.end()){
        transmission_ = beam_data["transmission"];
    }
    if (beam_data.find("sample_to_source_distance") != beam_data.end()){
        sample_to_source_distance_ = beam_data["sample_to_source_distance"];
    }
}

// serialize to json format
json Beam::to_json() const {
    // create a json object that conforms to a dials model serialization.
    json beam_data = {{"__id__", "monochromatic"}, {"probe", "x-ray"}};
    beam_data["wavelength"] = wavelength_;
    beam_data["direction"] = sample_to_source_direction_;
    beam_data["divergence"] = divergence_;
    beam_data["sigma_divergence"] =  sigma_divergence_;
    beam_data["polarization_normal"] = polarization_normal_;
    beam_data["polarization_fraction"] = polarization_fraction_;
    beam_data["flux"] = flux_;
    beam_data["transmission"] = transmission_;
    beam_data["sample_to_source_distance"] = sample_to_source_distance_;
    return beam_data;
}

double Beam::get_wavelength() const {
    return wavelength_;
}
void Beam::set_wavelength(double wavelength){
    wavelength_ = wavelength;
}

Vector3d Beam::get_s0() const {
    return -sample_to_source_direction_ / wavelength_;
}
void Beam::set_s0(Vector3d s0){
    double len = s0.norm();
    wavelength_ = 1.0 / len;
    sample_to_source_direction_ = -1.0 * s0 / len;
}

#endif //DX2_MODEL_BEAM_H