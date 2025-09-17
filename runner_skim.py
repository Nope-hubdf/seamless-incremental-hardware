import gc
import hashlib
import json
import os
import sys
import traceback as tb
from copy import deepcopy

import awkward as ak
import correctionlib
import hist
import numpy as np
import onnxruntime as ort
import spritz.framework.variation as variation_module
import uproot
import vector
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)
from spritz.modules.basic_selections import (
    LumiMask,
    lumi_mask,
    pass_flags,
    pass_trigger,
)
from spritz.modules.btag_sf import btag_sf
from spritz.modules.dnn_evaluator import dnn_evaluator, dnn_transform
from spritz.modules.gap_jet import gap_jet
from spritz.modules.jet_sel import clean_collection, jetSel
from spritz.modules.jme import (
    correct_jets_data,
    correct_jets_mc,
    jet_veto,
    remove_jets_HEM_issue,
)
from spritz.modules.lepton_sel import createLepton, leptonSel
from spritz.modules.lepton_sf import lepton_sf
from spritz.modules.prompt_gen import prompt_gen_match_leptons
from spritz.modules.puid_sf import puid_sf
from spritz.modules.puweight import puweight_sf
from spritz.modules.rochester import correctRochester, getRochester
from spritz.modules.run_assign import assign_run_period
from spritz.modules.theory_unc import theory_unc
from spritz.modules.trigger_sf import trigger_sf
from spritz.plugins.gen_analysis import genAnalysis

vector.register_awkward()

print("uproot version", uproot.__version__)
print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_puid = correctionlib.CorrectionSet.from_file(cfg["puidSF"])
ceval_btag = correctionlib.CorrectionSet.from_file(cfg["btagSF"])
ceval_puWeight = correctionlib.CorrectionSet.from_file(cfg["puWeights"])
ceval_lepton_sf = correctionlib.CorrectionSet.from_file(cfg["leptonSF"])
ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])
# ceval_fake = correctionlib.CorrectionSet.from_file(cfg["fakes"])

cset_trigger = correctionlib.CorrectionSet.from_file(cfg["triggerSF"])
# jec_stack = getJetCorrections(cfg)
rochester = getRochester(cfg)

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)
special_analysis_cfg = analysis_cfg["special_analysis_cfg"]
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1
dnn_cfg = special_analysis_cfg["dnn"]
onnx_session = ort.InferenceSession(dnn_cfg["model"], sess_opt)
dnn_t = dnn_transform(dnn_cfg["cumulative_signal"])


def ensure_not_none(arr):
    if ak.any(ak.is_none(arr)):
        raise Exception("There are some None in branch", arr[ak.is_none(arr)])
    return ak.fill_none(arr, -9999.9)


out_form = {
    "data": {
        "Lepton": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "pdgId",
                "isLoose",
                "isTight",
            ],
            "with_name": "Momentum4D",
        },
        "Jet": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "btagDeepFlavB",
            ],
            "with_name": "Momentum4D",
        },
        "PuppiMET": {
            "branches": [
                "pt",
                "phi",
            ],
            "with_name": None,
        },
        "PV": {
            "branches": [
                "npvsGood",
            ],
            "with_name": None,
        },
        "weight": {
            "branches": [],
            "with_name": None,
        },
    },
    "mc": {
        "Lepton": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "pdgId",
                "isLoose",
                "isTight",
                "promptgenmatched",
                "RecoSF",
                "TightSF",
            ],
            "with_name": "Momentum4D",
        },
        "Jet": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "btagDeepFlavB",
                "btagSF_deepjet_shape",
                "genJetIdx",
                "PUID_SF",
            ],
            "with_name": "Momentum4D",
        },
        "PuppiMET": {
            "branches": [
                "pt",
                "phi",
            ],
            "with_name": None,
        },
        "PV": {
            "branches": [
                "npvsGood",
            ],
            "with_name": None,
        },
        "GenPart": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "pdgId",
                "status",
                "genPartIdxMother",
                "statusFlags",
            ],
            "with_name": "Momentum4D",
        },
        "GenDressedLepton": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "pdgId",
            ],
            "with_name": "Momentum4D",
        },
        "GenJet": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
            ],
            "with_name": "Momentum4D",
        },
        "LHE": {
            "branches": [
                "NpNLO",
                "Njets",
                "Vpt",
            ],
            "with_name": None,
        },
        "LHEPart": {
            "branches": [
                "pt",
                "eta",
                "phi",
                "mass",
                "pdgId",
                "status",
            ],
            "with_name": "Momentum4D",
        },
        "prefireWeight": {
            "branches": [],
            "with_name": None,
        },
        "puWeight": {
            "branches": [],
            "with_name": None,
        },
        "TriggerSFweight_2l": {
            "branches": [],
            "with_name": None,
        },
        "weight": {
            "branches": [],
            "with_name": None,
        },
        "LHEScaleWeight": {
            "branches": [],
            "with_name": None,
        },
        "LHEPdfWeight": {
            "branches": [],
            "with_name": None,
        },
        "PSWeight": {
            "branches": [],
            "with_name": None,
        },
    },
}


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    trigger_sel = kwargs.get("trigger_sel", "")
    isData = kwargs.get("is_data", False)
    era = kwargs.get("era", None)
    subsamples = kwargs.get("subsamples", {})
    plugins = kwargs.get("plugins", [])
    special_weight = eval(kwargs.get("weight", "1.0"))

    output_variation_path = kwargs.get("output_variation_path", None)
    output_file_path = kwargs.get("output_file_path", None)

    # variations = {}
    # variations["nom"] = [()]
    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    if isData:
        events["weight"] = ak.ones_like(events.run)
    else:
        events["weight"] = events.genWeight

    if "EFT" in dataset:
        neft_rwgts = kwargs.get("nrwgts", 0)
        print(neft_rwgts)
        events = events[ak.num(events.LHEReweightingWeight) == neft_rwgts]
        events["rwgt"] = ak.pad_none(
            events.LHEReweightingWeight, neft_rwgts, clip=True, axis=1
        )
        events["rwgt"] = ak.fill_none(events.rwgt, 0.0)

    if isData:
        lumimask = LumiMask(cfg["lumiMask"])
        events = lumi_mask(events, lumimask)

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    # Add special weight for each dataset (not subsamples)
    if special_weight != 1.0:
        print(f"Using special weight for {dataset}: {special_weight}")

    events["weight"] = events.weight * special_weight

    # pass trigger and flags
    events = assign_run_period(events, isData, cfg, ceval_assign_run)
    events = pass_trigger(events, cfg["era"])
    events = pass_flags(events, cfg["flags"])

    events = events[events.pass_flags & events.pass_trigger]

    if isData:
        # each data DataSet has its own trigger_sel
        events = events[eval(trigger_sel)]

    # high pt muons
    events[("Muon", "pt")] = ak.where(
        events.Muon.pt > 200, events.Muon.pt * events.Muon.tunepRelPt, events.Muon.pt
    )

    events = jetSel(events, cfg)

    events = createLepton(events)

    events = leptonSel(events, cfg)
    # Latinos definitions, only consider loose leptons
    # remove events where ptl1 < 8
    events["Lepton"] = events.Lepton[events.Lepton.isLoose]
    # Apply a skim!
    events = events[ak.num(events.Lepton) >= 2]
    events = events[events.Lepton[:, 0].pt >= 8]

    if not isData:
        events = prompt_gen_match_leptons(events)

    print(events.Jet)
    # FIXME should clean from only tight / loose?
    # events = cleanJet(events)
    events = clean_collection(events, "Jet", "Lepton", 0.3)

    # # FatJet
    # events = clean_collection(events, "FatJet", "Lepton", 0.8)

    # tau2 = events.FatJet.tau_2
    # tau1 = events.FatJet.tau_1
    # tau21 = ak.fill_none(tau2 / ak.mask(tau1, tau1 != 0), 1.0)

    # events["FatJet"] = events.FatJet[
    #     (events.FatJet.pt > 200)
    #     & (events.FatJet.jetId >= 2)
    #     & (abs(events.FatJet.eta) < 2.4)
    #     & (tau21 <= 0.45)
    #     & (events.FatJet.msoftdrop > 40)
    #     & (events.FatJet.msoftdrop < 250)
    # ]

    # events = clean_collection(events, "Jet", "FatJet", 0.8)

    # Remove jets HEM issue
    events = remove_jets_HEM_issue(events, cfg)

    # # Jet veto maps
    events = jet_veto(events, cfg)

    eleWP = cfg["leptonsWP"]["eleWP"]
    muWP = cfg["leptonsWP"]["muWP"]

    events[("Lepton", "isTight")] = ak.fill_none(
        events.Lepton["isTightElectron_" + eleWP]
        | events.Lepton["isTightMuon_" + muWP],
        False,
    )

    res = {dataset: {"sumw": sumw, "nevents": nevents}}
    # res = {"done": True}

    # skim!

    # Require at least one good PV
    events = events[events.PV.npvsGood > 0]

    events = events[ak.num(events.Jet) >= 2]
    events = events[ak.num(events.Lepton[events.Lepton.isTight]) >= 2]

    print("Number of events after skim:", len(events))

    if len(events) == 0:
        return res

    if kwargs.get("top_pt_rwgt", False):
        top_particle_mask = (events.GenPart.pdgId == 6) & ak.values_astype(
            (events.GenPart.statusFlags >> 13) & 1, bool
        )
        toppt = ak.fill_none(
            ak.mask(events, ak.num(events.GenPart[top_particle_mask]) >= 1)
            .GenPart[top_particle_mask][:, -1]
            .pt,
            0.0,
        )

        atop_particle_mask = (events.GenPart.pdgId == -6) & ak.values_astype(
            (events.GenPart.statusFlags >> 13) & 1, bool
        )
        atoppt = ak.fill_none(
            ak.mask(events, ak.num(events.GenPart[atop_particle_mask]) >= 1)
            .GenPart[atop_particle_mask][:, -1]
            .pt,
            0.0,
        )

        top_pt_rwgt = (toppt * atoppt > 0.0) * (
            np.sqrt(np.exp(0.0615 - 0.0005 * toppt) * np.exp(0.0615 - 0.0005 * atoppt))
        ) + (toppt * atoppt <= 0.0)
        events["weight"] = events.weight * top_pt_rwgt

    # MCCorr
    # Should load SF and corrections here

    # # Correct Muons with rochester
    events = correctRochester(events, isData, rochester)

    if not isData:
        # puWeight
        events, variations = puweight_sf(events, variations, ceval_puWeight, cfg)

        # add trigger SF
        events, variations = trigger_sf(events, variations, cset_trigger, cfg)

        # add LeptonSF
        events, variations = lepton_sf(events, variations, ceval_lepton_sf, cfg)
        print(events.Jet)

        # FIXME add Electron Scale
        # FIXME add MET corrections?

        # Jets corrections

        # JEC + JER + JES
        events, variations = correct_jets_mc(
            events, variations, cfg, run_variations=False
        )

        # puId SF
        events, variations = puid_sf(events, variations, ceval_puid, cfg)

        # btag SF
        events, variations = btag_sf(events, variations, ceval_btag, cfg)

        # prefire

        if "L1PreFiringWeight" in ak.fields(events):
            events["prefireWeight"] = events.L1PreFiringWeight.Nom
            events["prefireWeight_up"] = events.L1PreFiringWeight.Up
            events["prefireWeight_down"] = events.L1PreFiringWeight.Dn
        else:
            events["prefireWeight"] = ak.ones_like(events.weight)
            events["prefireWeight_up"] = ak.ones_like(events.weight)
            events["prefireWeight_down"] = ak.ones_like(events.weight)

        variations.register_variation(
            columns=["prefireWeight"],
            variation_name="prefireWeight_up",
            format_rule=lambda _, var_name: var_name,
        )
        variations.register_variation(
            columns=["prefireWeight"],
            variation_name="prefireWeight_down",
            format_rule=lambda _, var_name: var_name,
        )
    else:
        events = correct_jets_data(events, cfg, era)
        # events = fake_evaluate(events, ceval_fake, cfg)

    # Fixes for None values

    assert not ak.any(ak.is_none(events.Lepton.pt, axis=1))
    events[("Lepton", "pt")] = ak.fill_none(events.Lepton.pt, 0.1)

    if not isData:
        btag_vars = [
            "lf",
            "hf",
            "hfstats1",
            "hfstats2",
            "lfstats1",
            "lfstats2",
            "cferr1",
            "cferr2",
            "jes",
        ]
        for var in btag_vars:
            assert not (
                ak.any(
                    ak.is_none(
                        events.Jet[f"btagSF_deepjet_shape_btag_{var}_up"], axis=1
                    )
                )
            )
            assert not (
                ak.any(
                    ak.is_none(
                        events.Jet[f"btagSF_deepjet_shape_btag_{var}_down"], axis=1
                    )
                )
            )
            events[("Jet", f"btagSF_deepjet_shape_btag_{var}_up")] = ak.fill_none(
                events[("Jet", f"btagSF_deepjet_shape_btag_{var}_up")],
                1.0,
            )
            events[("Jet", f"btagSF_deepjet_shape_btag_{var}_down")] = ak.fill_none(
                events[("Jet", f"btagSF_deepjet_shape_btag_{var}_down")],
                1.0,
            )

    masks = {}

    originalEvents = ak.copy(events)

    # loop over variations
    print("Doing variations")
    for variation in sorted(variations.get_variations_all()):
        events = ak.copy(originalEvents)

        if "JES" not in variation and "JER" not in variation and "nom" not in variation:
            continue

        switches = variations.get_variation_subs(variation)

        print(variation)
        for switch in switches:
            if len(switch) == 2:
                variation_dest, variation_source = switch
                events[variation_dest] = events[variation_source]

        masks[variation] = ak.ones_like(events.weight, dtype=bool)

        # resort Leptons
        lepton_sort = ak.argsort(events[("Lepton", "pt")], ascending=False, axis=1)
        events["Lepton"] = events.Lepton[lepton_sort]

        events["Lepton"] = ak.pad_none(events.Lepton, 2)

        events["mll"] = ak.fill_none(
            (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
            -9999.9,
        )

        # masks[variation] = masks[variation] & (abs(events.mll - 91.2) < 15.0)
        masks[variation] = masks[variation] & (events.mll > 50.0)

        jet_sort = ak.argsort(events[("Jet", "pt")], ascending=False, axis=1)
        events["Jet"] = events.Jet[jet_sort]

        events["Jet"] = events.Jet[events.Jet.pt >= 30]

        events["Jet"] = ak.pad_none(events.Jet, 2)

        # mjj cut
        events["mjj"] = ak.fill_none(
            (events.Jet[:, 0] + events.Jet[:, 1]).mass, -9999.9
        )
        masks[variation] = masks[variation] & (events.mjj > 200.0)

        # Define categories

        events["ee"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == -11 * 11
        events["mm"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == -13 * 13

        leptoncut = events.ee | events.mm

        # third lepton veto
        leptoncut = leptoncut & (
            ak.fill_none(
                ak.mask(
                    ak.all(events.Lepton[:, 2:].pt < 10, axis=1),
                    ak.num(events.Lepton) >= 3,
                ),
                True,
                axis=0,
            )
        )

        # Cut on pt of two leading leptons
        leptoncut = (
            leptoncut & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 13)
        )

        masks[variation] = masks[variation] & ak.fill_none(leptoncut, False)

    comb = 0
    for variation in sorted(masks):
        if isinstance(comb, int):
            comb = masks[variation]
        else:
            comb = comb | masks[variation]

    events = ak.copy(originalEvents)
    events = events[comb]

    print("Number of events after variations:", len(events))

    if not os.path.exists(output_variation_path) and not isData:
        write_chunks(variations, output_variation_path)

    # dump to root file
    if os.path.exists(output_file_path):
        raise Exception("Output file already exists", output_file_path)

    if len(events) == 0:
        return res

    def get_varied_cols(column):
        varied_cols = []
        for variation in sorted(variations.get_variations_all()):
            switches = variations.get_variation_subs(variation)

            for switch in switches:
                if len(switch) == 2:
                    variation_dest, variation_source = switch
                    if variation_dest == column:
                        varied_cols.append(variation_source)
        return varied_cols

    f = uproot.recreate(output_file_path)

    try:
        d = {}
        _form = out_form["data"] if isData else out_form["mc"]
        for field in _form:
            if field not in events.fields:
                print("did not find anything for", field, file=sys.stderr)
                continue
            if len(_form[field]["branches"]) == 0:
                d[field] = events[field]

                for field_varied in get_varied_cols(field):
                    d[field_varied] = events[field_varied]

            else:
                _d = {}
                for branch in _form[field]["branches"]:
                    col = (field, branch)

                    _d[branch] = events[col]

                    for field_varied in get_varied_cols(col):
                        _d[field_varied[-1]] = events[field_varied]
                d[field] = ak.zip(_d)

        f["Events"] = d
        f.close()

        res[output_file_path] = {
            "nevents": len(events),
        }

    except Exception as e:
        os.remove(output_file_path)
        raise e

    gc.collect()
    return res


if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    results = {}
    errors = []
    processed = []

    outfolder = analysis_cfg["outfolder"]
    output_variation_path = f"{outfolder}/variations.pkl"

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print(
                "Skip chunk",
                {k: v for k, v in new_chunk["data"].items() if k != "read_form"},
                "was already processed",
            )
            continue

        dataset = new_chunk["data"]["dataset"]
        print(dataset)
        fname = new_chunk["data"]["filenames"][0]

        prefix = "/store/data"
        if prefix not in fname:
            prefix = "/store/mc"
        fname = fname.split(prefix)[-1]

        start = new_chunk["data"]["start"]
        stop = new_chunk["data"]["stop"]

        unique_id = f"{dataset}_{fname}_{start}_{stop}"

        h = hashlib.new("sha256")
        h.update(unique_id.encode("utf-8"))
        unique_id = h.hexdigest()[:32]
        outfile = f"{outfolder}/{dataset}/FILE_{unique_id}.root"
        outfile = outfile.replace("//", "/")

        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        # if "DY-Pt-0" not in new_chunk["data"]["dataset"]:
        #     continue

        # # # FIXME run only on Zjj and DY
        # if new_chunk["data"]["dataset"] not in ["Zjj"]:
        #     continue

        # # FIXME run only on data
        # if not new_chunk["data"].get("is_data", False):
        #     continue

        # # FIXME process only one chunk per dataset
        # if new_chunk["data"]["dataset"] in processed:
        #     continue
        # processed.append(new_chunk["data"]["dataset"])

        # if "EFT" not in new_chunk["data"]["dataset"]:
        #     continue

        try:
            new_chunks[i]["result"] = big_process(
                process=process,
                output_variation_path=output_variation_path,
                output_file_path=outfile,
                **new_chunk["data"],
            )
            new_chunks[i]["error"] = ""
        except Exception as e:
            print("\n\nError for chunk", new_chunk, file=sys.stderr)
            nice_exception = "".join(tb.format_exception(None, e, e.__traceback__))
            print(nice_exception, file=sys.stderr)
            new_chunks[i]["result"] = {}
            new_chunks[i]["error"] = nice_exception

        print(f"Done {i + 1}/{len(new_chunks)}")

        # # FIXME run only on first chunk
        # if i >= 0:
        #     break

    # file = uproot.recreate("results.root")
    datasets = list(filter(lambda k: "root:/" not in k, results.keys()))
    # for dataset in datasets:
    #     print("Done", results[dataset]["nevents"], "events for dataset", dataset)
    #     file[dataset] = results[dataset]["events"]
    # file.close()

    # clean the events dictionary (too heavy and already saved in the root file)
    # for dataset in datasets:
    #     results[dataset]["events"] = {}

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
