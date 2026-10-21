
processList = {
    'wzp6_ee_nunuH_Huu_ecm240': {'fraction': 0.01},
    'wzp6_ee_nunuH_Hdd_ecm240': {'fraction': 0.01},
    'wzp6_ee_nunuH_Hss_ecm240': {'fraction': 0.01},
    'wzp6_ee_nunuH_Hcc_ecm240': {'fraction': 0.01},
    'wzp6_ee_nunuH_Hbb_ecm240': {'fraction': 0.01},
    'wzp6_ee_nunuH_Hgg_ecm240': {'fraction': 0.01},
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/pre_summer2026_training/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
# procDict = "FCCee_procDict_pre_summer2026_training_IDEA.json"
# Unfortunately, there is no procDict for pre_summer2026_training, but we have to have an existing file here even if we don't use the cross-section...
procDict = "FCCee_procDict_pre_summer2026_IDEA.json"

#Optional: output directory, default is local running directory
outputDir   = "outputs/jet-test"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS       = 6

# scale the histograms with the cross-section and integrated luminosity
doScale = False

# energy thresholds (GeV) for the inclusive ee_genkt jet multiplicity
INCL_EMIN = [0, 5]
# energy cut scan (GeV) for the inclusive jet multiplicity: first value, step, number of values
ECUT_SCAN = (0., 1., 61)

# Per-jet flavour categories (histograms <level>_<alg>_jets_category, one entry per jet). q is the Higgs decay
# flavour of the sample, a quark counts +1 and an antiquark -1.
#   0 clean q: q content +1, no other flavour     1 clean qbar: q content -1, no other flavour
#   2 no flavour: all contents 0                 3 q + other: q content +-1 plus another flavour
#   4 other: everything else                     5 single flavour (Hgg only): one flavour +-1, nothing else
# For get_flavour (parton_getflav_jets_category): 0 label q, 1 gluon, 2 unmatched, 3 other quark (any quark for Hgg).
FLAVOUR_INDEX = {"d": 0, "u": 1, "s": 2, "c": 3, "b": 4}
FLAVOUR_PDG = {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5}


def jet_category(name, qidx):
    """C++ expression for the per-jet flavour category of the jets {name}_jets (see above)."""
    contents = ", ".join(f"{name}_jets_{flv}_content[i]" for flv in FLAVOUR_INDEX)
    return f"""
        ROOT::VecOps::RVec<int> cat({name}_jets_b_content.size());
        for (size_t i = 0; i < cat.size(); ++i) {{
            const int fl[5] = {{{contents}}};
            const int q = {qidx};
            int n_flav = 0, n_other = 0, n_abs = 0;
            for (int k = 0; k < 5; ++k) {{
                n_flav += fl[k] != 0;
                n_other += k != q && fl[k] != 0;
                n_abs += std::abs(fl[k]);
            }}
            const int nq = q >= 0 ? fl[q] : 0;
            if (n_flav == 0) cat[i] = 2;
            else if (q >= 0 && nq == 1 && n_other == 0) cat[i] = 0;
            else if (q >= 0 && nq == -1 && n_other == 0) cat[i] = 1;
            else if (q >= 0 && std::abs(nq) == 1) cat[i] = 3;
            else if (q < 0 && n_abs == 1) cat[i] = 5;
            else cat[i] = 4;
        }}
        return cat;"""


# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):

    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    df = df.Alias("MCParticles", "Particle")

    # Higgs decay flavour of the sample, e.g. "bb" -> "b"; None for H -> gg
    higgs_decay = dataset.split("_H")[-1].split("_")[0]
    higgs_q = higgs_decay[0] if higgs_decay[0] in FLAVOUR_INDEX else None

    # count the Pythia special hadronisation entries
    hadronisation_pids = {
        # 88: "junction",
        91: "cluster",
        92: "string",
        93: "indep",  # independent fragmentation
    }
    df = df.Define("MC_pdg_all", "FCCAnalyses::MCParticle::get_pdg(MCParticles)")
    for pid, name in hadronisation_pids.items():
        df = df.Define(f"n_{name}_{pid}", f"Sum(MC_pdg_all == {pid})")
        results.append(df.Histo1D((f"n_{name}_{pid}", "", 10, 0., 10.), f"n_{name}_{pid}"))

    # final partons entering the hadronisation
    df = df.Define("HadronisationPartons", "FCCAnalyses::MCParticle::get_hadronisation_partons(MCParticles, _Particle_parents.index)")
    df = df.Define("n_hadronisation_partons", "HadronisationPartons.size()")
    results.append(df.Histo1D(("n_hadronisation_partons", "", 50, 0., 50.), "n_hadronisation_partons"))

    # primary hadrons, i.e. the hadrons directly after hadronisation, before they decay
    df = df.Define("PrimaryHadrons", "FCCAnalyses::MCParticle::get_primary_hadrons(MCParticles, _Particle_parents.index)")
    # flavoured pseudojets from the primary hadrons and, for comparison, from the hadronisation partons
    inputs = {
        "hadron_": "PrimaryHadrons",
        "parton_": "HadronisationPartons",
    }
    for prefix, particles in inputs.items():
        df = df.Define(f"{prefix}MC_px", f"FCCAnalyses::MCParticle::get_px({particles})")
        df = df.Define(f"{prefix}MC_py", f"FCCAnalyses::MCParticle::get_py({particles})")
        df = df.Define(f"{prefix}MC_pz", f"FCCAnalyses::MCParticle::get_pz({particles})")
        df = df.Define(f"{prefix}MC_m", f"FCCAnalyses::MCParticle::get_mass({particles})")
        df = df.Define(f"{prefix}MC_e", f"FCCAnalyses::MCParticle::get_e({particles})")
        df = df.Define(f"{prefix}MC_pdg", f"FCCAnalyses::MCParticle::get_pdg({particles})")
        df = df.Define(f"{prefix}pseudo_jets", f"FCCAnalyses::JetClusteringUtils::set_pseudoJets_xyzm_flav({prefix}MC_px, {prefix}MC_py, {prefix}MC_pz, {prefix}MC_m, {prefix}MC_pdg)")

        # 2-jet exclusive flavour-aware clustering, with IFN and with plain net flavour summation for comparison
        clusterings = {
            "ifn": f"JetClustering::flv_ifn_clustering_ee_kt(2, 2, 0, 0)({prefix}pseudo_jets)",
            "net": f"JetClustering::flv_net_clustering_ee_kt(2, 2, 0, 0)({prefix}pseudo_jets)",
        }

        df = df.Define(f"{prefix}MC_e_sum", f"Sum({prefix}MC_e)")
        results.append(df.Histo1D((f"{prefix}MC_e_sum", "", 250, 0., 250.), f"{prefix}MC_e_sum"))

        for algo, clustering in clusterings.items():
            name = f"{prefix}{algo}"
            df = df.Define(f"{name}_clustered_jets", clustering)

            df = df.Define(f"{name}_jets", f"FCCAnalyses::JetClusteringUtils::get_pseudoJets({name}_clustered_jets)")
            # df = df.Define(f"{name}_jetconstituents", f"FCCAnalyses::JetClusteringUtils::get_constituents({name}_clustered_jets)")
            for var in ["e", "px", "py", "pz", "phi", "m"]:
                df = df.Define(f"{name}_jets_{var}", f"FCCAnalyses::JetClusteringUtils::get_{var}({name}_jets)")

            for flv in ["b", "c", "s", "d", "u"]:
                df = df.Define(f"{name}_jets_{flv}_content", f"FCCAnalyses::JetClusteringUtils::get_{flv}_content({name}_jets)")

            # convert jets to LorentzVectors
            df = df.Define(f"{name}_jets_lv", f"Construct<ROOT::Math::PxPyPzEVector>({name}_jets_px, {name}_jets_py, {name}_jets_pz, {name}_jets_e)")
            df = df.Define(f"{name}_dijet_higgs_m", f"({name}_jets_lv[0]+{name}_jets_lv[1]).M()")

            # define histograms
            results.append(df.Histo1D((f"{name}_jets_e", "", 250, 0., 250.), f"{name}_jets_e"))
            results.append(df.Histo1D((f"{name}_M_jj", "", 250, 0., 250.), f"{name}_dijet_higgs_m"))
            for flv in ["b", "c", "s", "d", "u"]:
                results.append(df.Histo1D((f"{name}_jets_{flv}_content", "", 10, -5., 5.), f"{name}_jets_{flv}_content"))
            df = df.Define(f"{name}_jets_category", jet_category(name, FLAVOUR_INDEX.get(higgs_q, -1)))
            results.append(df.Histo1D((f"{name}_jets_category", "", 6, 0., 6.), f"{name}_jets_category"))

            # y_23 = d_23 / E_vis^2 (ee_kt distance at which the 3rd jet merges). IFN and net summation only
            # change the flavour, not the clustering sequence, so y_23 is the same for both.
            df = df.Define(f"{name}_y23", f"FCCAnalyses::JetClusteringUtils::get_exclusive_dmerge({name}_clustered_jets, 2) / ({prefix}MC_e_sum * {prefix}MC_e_sum)")
            # y_23 = 0 (fewer than 3 inputs) ends up in the underflow bin
            df = df.Define(f"{name}_log10_y23", f"std::log10({name}_y23)")

        # number of inclusive anti-kt jets (ee_genkt, R = 0.5, p = -1) above an energy threshold. The clustering
        # itself only supports a pT cut, so cluster without one and cut on the jet energy afterwards.
        df = df.Define(f"{prefix}incl_clustered_jets", f"JetClustering::clustering_ee_genkt(0.5, 0, 0, 0, 0, -1)({prefix}pseudo_jets)")
        df = df.Define(f"{prefix}incl_jets_e", f"FCCAnalyses::JetClusteringUtils::get_e({prefix}incl_clustered_jets.jets)")
        for emin in INCL_EMIN:
            df = df.Define(f"{prefix}incl_emin{emin}_njets", f"Sum({prefix}incl_jets_e > {emin})")
        # scan of the energy cut: number of jets with E > cut for every cut in ECUT_SCAN (filled into a 2D histogram)
        df = df.Define(f"{prefix}incl_ecut_njets", f"""
            ROOT::VecOps::RVec<float> n({ECUT_SCAN[2]});
            for (int i = 0; i < {ECUT_SCAN[2]}; ++i) n[i] = Sum({prefix}incl_jets_e > {ECUT_SCAN[0]} + i * {ECUT_SCAN[1]});
            return n;""")

    # For comparison, the standard FCCAnalyses jet flavour for the same parton-level jets: |PDG| of the
    # heaviest quark (else gluon = 21, unmatched = 0) within 0.3 rad. The Delphes samples only have
    # generator status 1 and 2, so all status 2 quarks and gluons of the event history are used for matching.
    df = df.Define("parton_getflav_jets_flavour", "FCCAnalyses::JetTaggingUtils::get_flavour(parton_ifn_jets, MCParticles)")
    results.append(df.Histo1D(("parton_getflav_jets_flavour", "", 23, 0., 23.), "parton_getflav_jets_flavour"))
    # same format as the flavour content histograms, but without sign: 1 if the jet has that flavour, else 0
    for flv, pdg in {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5}.items():
        df = df.Define(f"parton_getflav_jets_{flv}_content", f"ROOT::VecOps::RVec<int>(parton_getflav_jets_flavour == {pdg})")
        results.append(df.Histo1D((f"parton_getflav_jets_{flv}_content", "", 10, -5., 5.), f"parton_getflav_jets_{flv}_content"))
    getflav_q = FLAVOUR_PDG.get(higgs_q, -1)
    df = df.Define("parton_getflav_jets_category", f"""
        ROOT::VecOps::RVec<int> cat(parton_getflav_jets_flavour.size());
        for (size_t i = 0; i < cat.size(); ++i) {{
            const int l = parton_getflav_jets_flavour[i];
            cat[i] = l == {getflav_q} ? 0 : l == 21 ? 1 : l == 0 ? 2 : 3;
        }}
        return cat;""")
    results.append(df.Histo1D(("parton_getflav_jets_category", "", 4, 0., 4.), "parton_getflav_jets_category"))

    # Clean flavour signature at parton level (IFN): for H -> q qbar the jets carry q content +-1 and no
    # other flavour, for H -> gg neither jet carries any flavour. The two jets always have opposite flavour
    # content, so checking the leading jet is enough.
    expected = {flv: 1 if higgs_decay == flv * 2 else 0 for flv in ["b", "c", "s", "d", "u"]}
    clean = " && ".join(f"std::abs(parton_ifn_jets_{flv}_content[0]) == {n}" for flv, n in expected.items())
    df = df.Define("parton_ifn_clean", f"parton_ifn_jets_b_content.size() == 2 && {clean}")
    # x positions for the energy cut scan: the centre of the bin [cut, cut + step), safe against rounding
    df = df.Define("ecut_scan_values", f"ROOT::VecOps::RVec<float> x({ECUT_SCAN[2]}); for (int i = 0; i < {ECUT_SCAN[2]}; ++i) x[i] = {ECUT_SCAN[0]} + (i + 0.5) * {ECUT_SCAN[1]}; return x;")
    df_clean = df.Filter("parton_ifn_clean")
    df_unclean = df.Filter("!parton_ifn_clean")
    # y_23 of both clustering inputs, split by the parton-level flavour signature
    for name in ["parton_ifn", "hadron_ifn"]:
        results.append(df.Histo1D((f"{name}_log10_y23", "", 80, -8., 0.), f"{name}_log10_y23"))
        results.append(df_clean.Histo1D((f"{name}_log10_y23_clean", "", 80, -8., 0.), f"{name}_log10_y23"))
        results.append(df_unclean.Histo1D((f"{name}_log10_y23_unclean", "", 80, -8., 0.), f"{name}_log10_y23"))
    # number of inclusive ee_genkt jets, split by the same parton-level flavour signature
    for level in ["parton", "hadron"]:
        for emin in INCL_EMIN:
            name = f"{level}_incl_emin{emin}_njets"
            results.append(df.Histo1D((name, "", 30, 0., 30.), name))
            results.append(df_clean.Histo1D((f"{name}_clean", "", 30, 0., 30.), name))
            results.append(df_unclean.Histo1D((f"{name}_unclean", "", 30, 0., 30.), name))
        # x: energy cut, y: number of jets above it; one entry per event and cut value. The cut value is the
        # lower edge of its bin.
        name = f"{level}_incl_ecut_vs_njets"
        e0, de, ne = ECUT_SCAN
        model = (name, "", ne, e0, e0 + ne * de, 30, 0., 30.)
        for suffix, d in [("", df), ("_clean", df_clean), ("_unclean", df_unclean)]:
            results.append(d.Histo2D((model[0] + suffix, *model[1:]), "ecut_scan_values", f"{level}_incl_ecut_njets"))

    return results, weightsum

