# ---------------------------------------------------------------------------------
#                            Main Python file
# Desc: Run the models once the
from modules import Optimization_Outcome_Model as clv,outcome as outcome, survival as survival
import pandas


def execute_optimal_pricing_model():

    # Execute the pipeline

    # Query the database to load

    ####
    
    o_df = outcome.run()
    
    s_df = survival.run()

    clv_df = clv.run(o_df,s_df)


execute_optimal_pricing_model()




