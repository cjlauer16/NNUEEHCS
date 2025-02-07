import click
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def print_results(results, indent=0):
    for key, value in results.items():
        if isinstance(value, dict):
            print(' ' * indent, key)
            print_results(value, indent + 2)
        else:
            if value[3] == 'no_file':
                print(' ' * indent, key, value[3])
            else:
                print(' ' * indent, key, value[0], value[1], round(value[2], 2), round(value[3], 2))


def plot_quantity(results, quantity, output):
    # One file for each benchmark/dataset
    # Draw lines for the different methods and input directories
    for bench, datasets in results.items():
        for dataset, methods in datasets.items():
            fig, ax = plt.subplots()
            ax.set_title(f'{bench}/{dataset}')
            for method, method_data in sorted(methods.items()):
                for input_dir_name, data in method_data.items():
                    _, _, _, _, _, tresults = data
                    if tresults is None:
                        continue
                    label = f'{method} ({input_dir_name})'
                    ax.plot(tresults['trial'], tresults[quantity], label=label)
            ax.set_xlabel('Trial')
            ax.set_ylabel(quantity)
            ax.legend()
            fig.savefig(f'{output}/{bench}_{dataset}_{quantity}.png')
            plt.close(fig)


def barplot_quantity(results, quantity, output, statistic = 'max'):
    # One file for each benchmark/dataset
    # plot the maximum value of the quantity for each method
    for bench, datasets in results.items():
        for dataset, methods in datasets.items():
            fig, ax = plt.subplots()
            ax.set_title(f'{bench}/{dataset}')
            for method, method_data in sorted(methods.items()):
                for input_dir_name, data in method_data.items():
                    _, _, _, _, _, tresults = data
                    if tresults is None:
                        continue
                    label = f'{method} ({input_dir_name})'
                    if statistic == 'max':
                        ax.bar(label, tresults[quantity].max(), label=label)
                    elif statistic == 'median':
                        ax.bar(label, tresults[quantity].median(), label=label)
                    elif statistic == 'mean':
                        ax.bar(label, tresults[quantity].mean(), label=label)
            ax.set_xlabel('Method')
            ax.set_ylabel(quantity)
            ax.legend(loc='center right')
            ax.set_xticks([])
            fig.savefig(f'{output}/{bench}_{dataset}_{quantity}_{statistic}_bar.png')
            plt.close(fig)

def barplot_single_quantity(results, quantity, output):
    # One file for each benchmark/dataset
    # plot the maximum value of the quantity for each method
    for bench, datasets in results.items():
        for dataset, methods in datasets.items():
            fig, ax = plt.subplots()
            ax.set_title(f'{bench}/{dataset}')
            for method, method_data in sorted(methods.items()):
                for input_dir_name, data in method_data.items():
                    tresults = data[quantity]
                    if tresults is None:
                        continue
                    label = f'{method} ({input_dir_name})'
                    ax.bar(label, tresults, label=label)
            ax.set_xlabel('Method')
            ax.set_ylabel(quantity)
            ax.legend(loc='center right')
            ax.set_xticks([])
            fig.savefig(f'{output}/{bench}_{dataset}_{quantity}_bar.png')
            plt.close(fig)



# return the results for all method, but filtering all the results to only include the maximum value of the quantity
def narrow_to_statistic(results, quantity, statistic='max'):
    filtered_results = dict()
    for bench, datasets in results.items():
        if bench not in filtered_results:
            filtered_results[bench] = dict()
        for dataset, methods in datasets.items():
            if dataset not in filtered_results[bench]:
                filtered_results[bench][dataset] = dict()
            for method, method_data in sorted(methods.items()):
                if method not in filtered_results[bench][dataset]:
                    filtered_results[bench][dataset][method] = dict()
                for input_dir_name, data in method_data.items():
                    _, _, _, _, _, tresults = data
                    if tresults is None:
                        continue
                    if statistic == 'max':
                        max_value = tresults[quantity].max()
                    elif statistic == 'median':
                        max_value = tresults[quantity].median()
                    elif statistic == 'mean':
                        max_value = tresults[quantity].mean()
                    max_value_row = tresults[tresults[quantity] == max_value]
                    filtered_results[bench][dataset][method][input_dir_name] = max_value_row
    return filtered_results





def get_final_bo_trial(result_files):
    import re
    number_re = re.compile(r'\d+')
    largest_trial = None
    largest_trial_num = None
    for filename in result_files:
        fname = str(filename.name)
        trial_num = int(number_re.search(fname).group())
        file_exists = Path(f'{filename}/trial_results.csv').exists()
        if (largest_trial is None or trial_num > largest_trial_num) and file_exists:
            largest_trial = filename
            largest_trial_num = trial_num
    return largest_trial_num, largest_trial


@click.command()
@click.option('--input', 'inputs', type=click.Path(exists=True), multiple=True, help='Path to directory containing experiment results')
@click.option('--output', type=click.Path(), help='Path to output file')
def main(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    results = dict()

    for input_dir in inputs:
        input_dir = Path(input_dir)
        input_dir_name = input_dir.name
        for benchdir in input_dir.iterdir():
            # skip if not a directory
            if not benchdir.is_dir():
                continue
            bench_name = benchdir.name
            if bench_name not in results:
                results[bench_name] = dict()
            bench_results = results[bench_name]
            for datasetdir in benchdir.iterdir():
                dataset_name = datasetdir.name
                if dataset_name not in bench_results:
                    bench_results[dataset_name] = dict()
                dataset_results = bench_results[dataset_name]
                for methoddir in datasetdir.iterdir():
                    method_name = methoddir.name
                    if method_name not in dataset_results:
                        dataset_results[method_name] = dict()
                    method_results = dataset_results[method_name]

                    bo_files = list()
                    for resultfile in methoddir.iterdir():
                        bo_files.append(resultfile)
                    trial_num, trial_file = get_final_bo_trial(bo_files)
                    if trial_file is None:
                        method_results[input_dir_name] = (0, 0, 0, "no_file", None)
                        continue
                    tresults = pd.read_csv(f'{trial_file}/trial_results.csv')
                    try:
                        max_uedist = tresults['wasserstein_distance'].argmax()
                        max_uedist_value = tresults['wasserstein_distance'].max()
                        median_uedist_value = tresults['wasserstein_distance'].median()
                    except KeyError:
                        try:
                            max_uedist = tresults['jensen_shannnon_distance'].argmax()
                            max_uedist_value = tresults['jensen_shannon_distance'].max()
                            median_uedist_value = tresults['jensen_shannon_distance'].median()
                        except KeyError:
                            try:
                                max_uedist = tresults['sensitivity'].argmax()
                                max_uedist_value = tresults['sensitivity'].max()
                                median_uedist_value = tresults['sensitivity'].median()
                            except KeyError:
                                max_uedist = tresults['score_dist'].argmax()
                                max_uedist_value = tresults['score_dist'].max()
                                median_uedist_value = tresults['score_dist'].median()


                    method_results[input_dir_name] = (trial_num, max_uedist,
                                                      max_uedist_value,
                                                      median_uedist_value,
                                                      trial_file,
                                                      tresults
                                                      )

    print_results(results)
    plot_quantity(results, 'sensitivity', output)
    barplot_quantity(results, 'sensitivity', output)
    barplot_quantity(results, 'sensitivity', output, statistic='median')
    plot_quantity(results, 'ue_time', output)

    filtered_results = narrow_to_statistic(results, 'sensitivity')
    barplot_single_quantity(filtered_results, 'ue_time', output)


if __name__ == '__main__':
    main()
