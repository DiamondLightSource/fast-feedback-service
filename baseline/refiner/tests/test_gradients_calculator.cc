#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>
#include <dx2/detector.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "detector_parameterisation.cc"
#include "beam_parameterisation.cc"
#include "cell_parameterisation.cc"
#include "orientation_parameterisation.cc"
#include "gemmi/symmetry.hpp"
#include "gemmi/unitcell.hpp"
#include "gradients_calculator.cc"
#include <dx2/reflection.hpp>

using Eigen::Vector3d;

TEST(BaselineIndexer, gradients_calculator_test) {
    std::string imported_expt = "/dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/imported.expt";
    std::ifstream f(imported_expt);
    json elist_json_obj = json::parse(f);
    json detector_json = elist_json_obj["detector"][0];
    Detector detector(detector_json);
    Panel panel = detector.panels()[0];
    json beam_json = elist_json_obj["beam"][0];
    json goniometer_json = elist_json_obj["goniometer"][0];
    MonochromaticBeam beam(beam_json);
    Goniometer goniometer(goniometer_json);
    
    BeamParameterisation beam_param(beam, goniometer);
    DetectorParameterisation detector_param(panel);
    
    // Set non-zero parameters
    std::vector<double> new_beam_params = {1.0,2.0, 0.90};
    beam_param.set_params(new_beam_params);
    std::vector<double> new_det_params = {169.0,-6.8,7.0,6,5,3};
    detector_param.set_params(new_det_params);

    Vector3d a = {-0.19,2.78,6.05};
    Vector3d b = {0.09,-15.44,8.78};
    Vector3d c = {26.77,0.45,2.39};
    gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
    Crystal crystal(a,b,c, space_group);
    OrientationParameterisation orientation_parameterisation(crystal);
    CellParameterisation cell_parameterisation(crystal);

    // Set non-zero parameters
    std::vector<double> new_orient_params = {1.0,2.0, 20};
    std::vector<double> new_cell_params = {2279,320,138,-71,-31,-6};
    orientation_parameterisation.set_params(new_orient_params);
    cell_parameterisation.set_params(new_cell_params);

    GradientsCalculator calculator(orientation_parameterisation, cell_parameterisation,
        goniometer, beam_param, detector_param);

    // The calculator is validated against the equivalent calculator results in dials,
    // on data from a couple of randomly chosen reflections.
    ReflectionTable table;
    std::vector<int> miller_index = {-12,-11,17,-2,6,14};
    std::vector<double> s1 = {0.4179348780141723, -0.20045075910012414, -0.6600541717187897, -0.09066552735670198, -0.2355641335951222, -0.7660358214865836};
    std::vector<double> xyzcal = {327.2532679418043, 260.486750141457, 2.8784005884739994, 198.93046946782965, 261.3879205849422, 2.8810375748248247};

    table.add_column("miller_index", 2, 3, miller_index);
    table.add_column("s1", 2, 3, s1);
    table.add_column("xyzcal.mm", 2, 3, xyzcal);
    // Order of gradients here - beam, orientation, cell, detector
    std::vector<std::vector<double>> gradients = calculator.get_gradients(table);
    
    // Beam parameters 0 and 2 are fixed - so zero gradients are returned.
    std::vector<std::vector<double>> expected_beam_gradients;
    expected_beam_gradients.push_back({0.0,0.0,0.0,0.0,0.0,0.0});
    expected_beam_gradients.push_back({0.4287826631965805, 0.18408783635453912, 0.20191156388350756, -0.17853420664143882, -0.0008082088205582376, -0.002344544175499804});
    expected_beam_gradients.push_back({0.0,0.0,0.0,0.0,0.0,0.0});
    for (int i=0;i<3;++i){
        for (int j=0;j<gradients[i].size();++j){
            EXPECT_NEAR(gradients[i][j], expected_beam_gradients[i][j], 1E-12);
        }
    }
    std::vector<std::vector<double>> expected_orientation_gradients;
    expected_orientation_gradients.push_back({-0.04959579644558594, 0.025070920084672176, -0.12220349700104796, -0.06820402967379478, -0.0007634711810092794, -0.0015109861808262255});
    expected_orientation_gradients.push_back({-0.5062005276059334, 0.00469820527228704, 0.20986412961046694, -0.041038216518272025, -1.76482882093943E-5, -0.0016294842009530364});
    expected_orientation_gradients.push_back({0.06494068374443888, 0.10459812784210254, -0.08929501703416928, -0.16857128301109173, -0.0003305266605921589, -0.0007779020610478103});
    int offset = 3;
    for (int i=0;i<3;++i){
        for (int j=0;j<gradients[i+offset].size();++j){
            EXPECT_NEAR(gradients[i+offset][j], expected_orientation_gradients[i][j], 1E-12);
        }
    }
    std::vector<std::vector<double>> expected_cell_gradients;
    expected_cell_gradients.push_back({0.044929397934425894, 0.0032113606061801863, 0.12239571338813976, 0.006501737891694474, -0.00010782518902494864, -9.496308786352764E-5});
    expected_cell_gradients.push_back({-0.0009982417164380055, -0.016059493329712186, 0.021925146205354643, 0.17738128609561624, -0.0006094409461294244, 0.0016907180188751464});
    expected_cell_gradients.push_back({0.5920492522192147, 0.40227621146774833, 0.007649280448604453, 0.11045187349562537, -0.0003585427716161955, 0.0024151582447524337});
    expected_cell_gradients.push_back({0.01456038676210086, 0.013151546827410026, 0.08278997918317386, -0.12471862951189557, -0.0011055814255902538, -0.0012655746686363915});
    expected_cell_gradients.push_back({-1.0027102548270417, -0.11441398647948181, 0.057507694508682414, -0.03465028767685144, 0.0004956756740565934, -0.0008011897845487581});
    expected_cell_gradients.push_back({-0.9350508965438956, 0.3508435894916053, 0.01951313914075557, 0.08385438357345058, 0.0002414885505191235, 0.0019732238690136836});
    offset = 6;
    for (int i=0;i<6;++i){
        for (int j=0;j<gradients[i+offset].size();++j){
            EXPECT_NEAR(gradients[i+offset][j], expected_cell_gradients[i][j], 1E-12);
        }
    }
    std::vector<std::vector<double>> expected_detector_gradients;
    expected_detector_gradients.push_back({0.6347498332644728, -0.11673565064776006, 0.29977337029334195, 0.30880352915599546, 0.0, 0.0});
    expected_detector_gradients.push_back({-1.0, -1.0, 0.0, 0.0, 0.0, 0.0});
    expected_detector_gradients.push_back({0.0, 0.0, -1.0, -1.00, 0.0, 0.0});
    expected_detector_gradients.push_back({0.05066169957957481, 0.05218779642736319, -0.10727272182169596, 0.01972832495947142, 0.0, 0.0});
    expected_detector_gradients.push_back({0.032413652010338245, -0.006232995148877551, 0.015637403503896467, 0.016340483302854734, 0.0, 0.0});
    expected_detector_gradients.push_back({-0.06783309907395624, -0.0026599798137931815, -0.03206633548057176, 0.0061907114140071235, 0.0, 0.0});
    offset = 12;
    for (int i=0;i<6;++i){
        for (int j=0;j<gradients[i+offset].size();++j){
            EXPECT_NEAR(gradients[i+offset][j], expected_detector_gradients[i][j], 1E-12);
        }
    }
}