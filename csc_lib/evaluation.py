from tqdm import tqdm
from csc_lib.config import GEN_ARGS

def measure_chances_generating_target(generator, df, target_column, prompt_template, nb_tries, max_new_tokens, step=1, block_size=1):
    """
    Iterates through rows of a dataframe and keeps track of how many information from a target column we can obtain when using prompt_template in the limit of nb_tries tries.
    
    Example usage :
    measure_chances_generating_target(
        generator = generator,
        df = df,
        target_column = 'annotations',
        prompt_template = 'Profil:\nage{age} ; sexe : {sexe} ; lexique: {indication}\nCas clinique:\n',
        nb_tries = 30,
        max_new_tokens = 1300)
    """

    # Check if the target column is in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"The column '{target_column}' is not present in the dataframe.")

    df[f'# {target_column} found each try'] = None
    df[f'# {target_column} found each try'] = df[f'# {target_column} found each try'].astype(object)
    df[f'Max {target_column} found'] = None
    df[f'Max {target_column} found'] = df[f'Max {target_column} found'].astype(object)

    for bloc_idx in tqdm(range(0, len(df)-block_size, step), desc='Iterating through blocks...'):

        for i, row in df[bloc_idx:bloc_idx + block_size].iterrows():

            # prompt = prompt_template.format(**row)
            antecedents = ', '.join(row['antecedents'])
            prompt = prompt_template.format(antecedents=antecedents)
            targets = row[target_column]
            nb_targets_found_over_each_try = []
            max_targets_found = []

            for _ in range(nb_tries):

                completed_prompt = generator.complete_prompt(prompt, GEN_ARGS, max_new_tokens)
                generated_part = completed_prompt[len(completed_prompt):] # we exclude the prompt
                found_targets = {target for target in targets if generated_part.find(target)!=-1}
                nb_targets_found = len(found_targets)
                nb_targets_found_over_each_try.append(nb_targets_found)

                if nb_targets_found >= len(max_targets_found):
                    max_targets_found = found_targets

            df.at[i, f'# {target_column} found each try'] = nb_targets_found_over_each_try
            df.at[i, f'Max {target_column} found'] = max_targets_found

    return df