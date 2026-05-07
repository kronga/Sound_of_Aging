"""
Run volcano analysis for all biological modalities using existing multi-seed LGBM predictions,
then run for voice using the new Ridge predictions from step 3.
"""
import sys
import os
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import volcano_visualization as vv

SUBJECT_DETAILS_CSV = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/subject_details_df_Oct25.csv"


def _enrich_voice_predictions(pred_csv: str, out_dir: str) -> str:
    """
    Voice predictions lack research_stage. Join subject_details on filename
    (index col) to add it (visit_number has the same values as research_stage).
    """
    pred = pd.read_csv(pred_csv)
    # pred has 'index' = filename, 'group' = subject_number
    sd = pd.read_csv(SUBJECT_DETAILS_CSV, usecols=["filename", "visit_number"])
    sd = sd.rename(columns={"visit_number": "research_stage"})
    pred = pred.merge(sd, left_on="index", right_on="filename", how="left")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions_averaged_enriched.csv")
    pred.to_csv(out_path, index=False)
    return out_path


def _voice_template_enriched(gender: str, out_dir: str) -> str:
    src = os.path.join(VOICE_RIDGE_ROOT, f"gender_{gender}", "predictions_averaged.csv")
    return _enrich_voice_predictions(src, os.path.join(out_dir, f"gender_{gender}"))

LGBM_ROOT = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/Oct25_voice_full_length/LGBM_stuff_new"
VOICE_RIDGE_ROOT = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step3_voice_age_ridge"
OUTPUT_ROOT = "/home/davidkro/PycharmProjects/DeepVoice/paper_revision_outputs/step5_volcano"

BIOLOGICAL_MODALITIES = [
    "blood_test",
    "DEXA",
    "diet",
    "metabolomics",
    "microbiome",
    "NMR",
    "retina",
    "sleep",
]


def run_modality(modality: str, predictions_path_template: str) -> None:
    print(f"\n{'='*60}")
    print(f"VOLCANO: {modality.upper()}")
    print(f"{'='*60}")
    vv.MODALITY = modality
    vv.BASE_PREDICTIONS_PATH = predictions_path_template
    vv.OUTDIR = os.path.join(OUTPUT_ROOT, modality)
    os.makedirs(vv.OUTDIR, exist_ok=True)
    try:
        vv.main()
    except Exception as e:
        print(f"[ERROR] {modality}: {e}")


if __name__ == "__main__":
    # Biological modalities — use existing LGBM_stuff_new predictions
    lgbm_template = os.path.join(
        LGBM_ROOT,
        "multi_seed_lgbm_age_prediction_{modality}",
        "gender_{gender}",
        "predictions_averaged.csv",
    )
    for mod in BIOLOGICAL_MODALITIES:
        pred_path = lgbm_template.replace("{modality}", mod)
        # Quick check that at least one gender exists
        male_path = pred_path.replace("{gender}", "male")
        if not os.path.exists(male_path):
            print(f"[SKIP] {mod}: predictions not found at {male_path}")
            continue
        run_modality(mod, lgbm_template)

    # Voice — enrich predictions with research_stage then run
    voice_male_path = os.path.join(VOICE_RIDGE_ROOT, "gender_male", "predictions_averaged.csv")
    if os.path.exists(voice_male_path):
        print(f"\n{'='*60}\nVOLCANO: VOICE\n{'='*60}")
        voice_out = os.path.join(OUTPUT_ROOT, "voice")
        os.makedirs(voice_out, exist_ok=True)
        # Pre-enrich each gender file with research_stage and write enriched copies
        for gender in ["male", "female"]:
            src = os.path.join(VOICE_RIDGE_ROOT, f"gender_{gender}", "predictions_averaged.csv")
            os.makedirs(os.path.join(voice_out, f"gender_{gender}"), exist_ok=True)
            _enrich_voice_predictions(src, os.path.join(voice_out, f"gender_{gender}"))
        enriched_template = os.path.join(voice_out, "gender_{gender}", "predictions_averaged_enriched.csv")
        run_modality("voice", enriched_template)
    else:
        print(f"\n[SKIP] voice: step-3 predictions not yet available at {voice_male_path}")
        print("Re-run this script after step 3 (run_age_prediction.py) completes.")

    print("\nAll volcano runs complete.")
