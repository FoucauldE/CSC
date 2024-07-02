import os
import pandas as pd
import argparse
from csc_lib.data_loader import load_models
from csc_lib.data_processing import correct_literal_eval
from csc_lib.generation import Generator
from csc_lib.evaluation import measure_chances_generating_target
from csc_lib.visualize import violin_plot
from csc_lib.config import FT_MODEL_PATH, OUTPUT_PATH

def main(association_rules_path, experiment_name, nb_tries, max_new_tokens, step, block_size, ft_model_path=FT_MODEL_PATH):

    # Check if association rules path exists
    if not os.path.isfile(association_rules_path):
        raise FileNotFoundError(f"The specified path for association rules does not exists: {association_rules_path}")
    
    # Load association rules
    df_rules = pd.read_csv(association_rules_path, index_col=0)
    df_rules = correct_literal_eval(df_rules, ['antecedents', 'consequents'])

    # Prepare fine-tuned model to complete new prompts
    tokenizer, base_model, ft_model = load_models(ft_model_path)
    generator = Generator(tokenizer, ft_model)
    # prompt_template = 'Profil:\nage{age} ; sexe : {sexe} ; lexique: {indication}\nCas clinique:\n'
    # prompt_template = "<|startoftext|> lexique: {', '.join(antecedents)},"
    prompt_template = "<|startoftext|> lexique: {antecedents},"

    # Measure chances that our fine-tuned model generates consequents when prompting antecedents
    df_measures = measure_chances_generating_target(
        generator=generator,
        df=df_rules,
        target_column='consequents',
        prompt_template=prompt_template,
        nb_tries=nb_tries,
        max_new_tokens=max_new_tokens,
        step=step,
        block_size=block_size
        )
    
    # Plot the results
    violin_plot_figure = violin_plot(
        df_measures,
        'consequents',
        'Violin Plots of consequents founds for each prompted antecedents\n(over 30 tries)'
    )
    
    # Save the results
    save_path = os.path.join(OUTPUT_PATH, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    df_measures.to_csv(os.path.join(save_path,'measures.csv'))
    violin_plot_figure.savefig(os.path.join(save_path, 'violin_plot.png'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Measure to which extent a model can generate consequents when prompting antecedents')
    parser.add_argument('-p', '--path_rules', type=str, required=True, help='Path to the association rules csv file to study')
    parser.add_argument('-ft', '--ft_model_path', type=str, default=FT_MODEL_PATH, help='Path to the fine-tuned model')
    parser.add_argument('-n', '--number_tries', type=int, default=30, help='Number of generations for each studied case.')
    parser.add_argument('-t', '--max_tokens', type=int, default=300, help='Number of new tokens.')
    parser.add_argument('-s', '--step', type=int, default=1, help='Step size between blocks.')
    parser.add_argument('-b', '--block_size', type=int, default=1, help='Block size')
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    args = parser.parse_args()

    association_rules_path, experiment_name, nb_tries, max_new_tokens, step, block_size, ft_model_path, = args.path_rules, args.experiment_name, args.number_tries, args.max_tokens, args.step, args.block_size, args.ft_model_path

    main(association_rules_path, experiment_name, nb_tries, max_new_tokens, step, block_size, ft_model_path)