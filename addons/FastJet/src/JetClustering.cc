#include "FastJet/JetClustering.h"
#include "FastJet/ExternalRecombiner.h"

#include "FCCAnalyses/JetClusteringUtils.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"

#include <iostream>
#include <stdexcept>
#include <string>

// #include "fastjet/contrib/FlavInfo.hh"

namespace {
  // IFN only neutralises flavour, the kinematic clustering sequence is the one
  // of the base algorithm, so exclusive jets are as meaningful as for ee_kt.
  // Declaring this also avoids the ClusterSequence "dcut and exclusive jets"
  // LimitedWarning. Any LimitedWarning must be avoided: libDelphes bundles its
  // own FastJet, so the global warnings summary list gets destroyed twice at
  // exit (once by libfastjet, once by libDelphes) -> double free if non-empty.
  class ExclusiveIFNPlugin : public fastjet::contrib::IFNPlugin {
  public:
    using fastjet::contrib::IFNPlugin::IFNPlugin;
    bool exclusive_sequence_meaningful() const override { return true; }
  };

  // The FastJet contrib FlavRecombiner only does E-scheme kinematics, so any
  // other recombination scheme would be silently ignored. Refuse it instead.
  void check_flav_recombination(int recombination, const std::string& name) {
    if (recombination == 0)
      return;
    const std::string msg = "JetClustering::" + name + ": recombination scheme " + std::to_string(recombination) +
                            " is not supported for flavour-aware clustering, only the E scheme (0) is.";
    std::cerr << "ERROR: " << msg << std::endl;
    throw std::invalid_argument(msg);
  }

  // ee_kt with IFN: alpha = 2, omega = 3 - alpha, net flavour summation.
  fastjet::JetDefinition ifn_ee_kt_definition() {
    fastjet::JetDefinition base_jet_def(fastjet::JetAlgorithm::ee_kt_algorithm);
    base_jet_def.set_recombiner(new fastjet::contrib::FlavRecombiner(fastjet::contrib::FlavRecombiner::net));
    // must be set before the plugin copies base_jet_def, so the copies share ownership
    base_jet_def.delete_recombiner_when_unused();

    // The main free parameter, alpha, in the uij distance,
    //   uij = max(pt_i, pt_j)^alpha min(pt_i, pt_j)^(2-alpha) Omega_ij
    double alpha = 2.0;

    // The parameter that sets the nature of the Omega rapidity term;
    // only change the default of 3-alpha if you are sure you know what you are doing
    double omega = 3.0 - alpha;

    auto* plugin = new ExclusiveIFNPlugin(base_jet_def, alpha, omega, fastjet::contrib::FlavRecombiner::net);
    fastjet::JetDefinition def(plugin);
    def.delete_plugin_when_unused();
    return def;
  }

  // IFN can neutralise the flavour of a jet that is not itself merged at a later step (the jet i that
  // gets merged is neutralised against a third jet k). The FlavHistory of exclusive jets is shared with
  // the cluster sequence and therefore holds the flavour at the end of the full clustering. Replace it
  // by the flavour after the last merging step that was actually performed for these exclusive jets.
  void freeze_exclusive_flavours(std::vector<fastjet::PseudoJet>& jets, unsigned int n_inputs) {
    if (jets.empty())
      return;
    // history steps 0 .. n_inputs - 1 are the inputs and every merging adds one step, so the n exclusive
    // jets exist after step 2 * n_inputs - n - 1
    const int last_step = 2 * n_inputs - jets.size() - 1;
    for (auto& jet : jets) {
      auto* frozen = new fastjet::contrib::FlavHistory(
          jet.user_info<fastjet::contrib::FlavHistory>().flavour_at_step(last_step), last_step);
      jet.set_user_info(frozen);
    }
  }
}  // namespace

namespace JetClustering {

  clustering_kt::clustering_kt(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::kt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _radius, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_kt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    FCCAnalysesJet result = FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);

    return result;
  }

  clustering_antikt::clustering_antikt(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    //fastjet::JetAlgorithm jetAlgorithm{fastjet::JetAlgorithm::undefined_jet_algorithm};
    _jetAlgorithm = fastjet::JetAlgorithm::antikt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _radius, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_antikt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    FCCAnalysesJet result = FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);

    return result;
  }

  clustering_cambridge::clustering_cambridge(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::cambridge_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _radius, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_cambridge::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  clustering_ee_kt::clustering_ee_kt(int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::ee_kt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_ee_kt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  clustering_ee_genkt::clustering_ee_genkt(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination, float arg_exponent) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;
    _exponent = arg_exponent;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::ee_genkt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _radius, _exponent, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_ee_genkt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  clustering_genkt::clustering_genkt(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination, float arg_exponent) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;
    _exponent = arg_exponent;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::genkt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm, _radius, _exponent, _recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_genkt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);

    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  clustering_valencia::clustering_valencia(float arg_radius,
                                           int arg_exclusive,
                                           float arg_cut,
                                           int arg_sorted,
                                           int arg_recombination,
                                           float arg_beta,
                                           float arg_gamma) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;
    _beta = arg_beta;
    _gamma = arg_gamma;

    // initialize jet algorithm
    _jetAlgorithm = new fastjet::contrib::ValenciaPlugin(_radius, _beta, _gamma);

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    // define the clustering sequence and jet definition
    _def = fastjet::JetDefinition(_jetAlgorithm);
    _def.set_recombination_scheme(_recombScheme);
    if (_recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_valencia::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  clustering_jade::clustering_jade(
      float arg_radius, int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    _radius = arg_radius;
    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = new fastjet::JadePlugin();

    // initialize recombination scheme
    fastjet::RecombinationScheme recombScheme =
        FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    _def = fastjet::JetDefinition(_jetAlgorithm);
    _def.set_recombination_scheme(recombScheme);
    if (recombScheme == fastjet::RecombinationScheme::external_scheme)
      _def.set_recombiner(new ExternalRecombiner(_recombination));
  }

  FCCAnalysesJet clustering_jade::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);

    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  flv_ifn_clustering_ee_kt::flv_ifn_clustering_ee_kt(int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    check_flav_recombination(arg_recombination, "flv_ifn_clustering_ee_kt");

    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::ee_kt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    _def = ifn_ee_kt_definition();
  }

  FCCAnalysesJet flv_ifn_clustering_ee_kt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    if (_exclusive != 0)
      freeze_exclusive_flavours(pjets, input.size());
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

  flv_ifn_ycut_scan_ee_kt::flv_ifn_ycut_scan_ee_kt(const std::vector<float>& arg_ycuts, int arg_sorted) {
    _ycuts = arg_ycuts;
    _sorted = arg_sorted;
    _def = ifn_ee_kt_definition();
  }

  std::vector<FCCAnalysesJet> flv_ifn_ycut_scan_ee_kt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    std::vector<FCCAnalysesJet> result;
    if (input.empty()) {
      result.resize(_ycuts.size(), FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet());
      return result;
    }

    // a single clustering: the sequence (and with frozen flavours the IFN flavour) at a given ycut does not
    // depend on the later merging steps
    _cs = fastjet::ClusterSequence(input, _def);

    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    for (float ycut : _ycuts) {
      std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, 4, ycut, _sorted);
      freeze_exclusive_flavours(pjets, input.size());
      result.push_back(FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max));
    }
    return result;
  }

  flv_net_clustering_ee_kt::flv_net_clustering_ee_kt(int arg_exclusive, float arg_cut, int arg_sorted, int arg_recombination) {
    check_flav_recombination(arg_recombination, "flv_net_clustering_ee_kt");

    _exclusive = arg_exclusive;
    _cut = arg_cut;
    _sorted = arg_sorted;
    _recombination = arg_recombination;

    // initialize jet algorithm
    _jetAlgorithm = fastjet::JetAlgorithm::ee_kt_algorithm;

    // initialize recombination scheme
    _recombScheme = FCCAnalyses::JetClusteringUtils::recomb_scheme(_recombination);

    _def = fastjet::JetDefinition(_jetAlgorithm);
    _def.set_recombiner(new fastjet::contrib::FlavRecombiner(fastjet::contrib::FlavRecombiner::net));
    _def.delete_recombiner_when_unused();
  }

  FCCAnalysesJet flv_net_clustering_ee_kt::operator()(const std::vector<fastjet::PseudoJet>& input) {
    //return empty struct
    if (FCCAnalyses::JetClusteringUtils::check(input.size(), _exclusive, _cut) == false)
      return FCCAnalyses::JetClusteringUtils::initialise_FCCAnalysesJet();

    _cs = fastjet::ClusterSequence(input, _def);

    //cluster jets
    std::vector<fastjet::PseudoJet> pjets = FCCAnalyses::JetClusteringUtils::build_jets(_cs, _exclusive, _cut, _sorted);
    //get dmerged elements
    std::vector<float> dmerge = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 0);
    std::vector<float> dmerge_max = FCCAnalyses::JetClusteringUtils::exclusive_dmerge(_cs, 1);

    //transform to FCCAnalysesJet
    return FCCAnalyses::JetClusteringUtils::build_FCCAnalysesJet(pjets, dmerge, dmerge_max);
  }

}  // namespace JetClustering
