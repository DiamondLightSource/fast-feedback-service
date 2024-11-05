#ifndef DX2_MODEL_BEAM_H
#define DX2_MODEL_BEAM_H
#include <Eigen/Dense>
using Eigen::Vector3d;

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

// full constructor for to-from json
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