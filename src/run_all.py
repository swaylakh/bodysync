"""Run the full BodySync pipeline end to end: python src/run_all.py"""
import clean
import explain
import explore
import features
import model
import model_v2
import model_v3
import sync_score
import validate_sleep
import wesad
import bodysync_validate

if __name__ == "__main__":
    clean.clean()
    features.main()
    explore.main()
    model.main()
    model_v2.main()
    model_v3.main()
    explain.shap_plot()
    explain.body_weather_map()
    explain.desync_analysis()
    sync_score.main()
    validate_sleep.main()
    wesad.main()          # needs data/wesad/WESAD.zip
    bodysync_validate.wesad()
    bodysync_validate.zupwd()
