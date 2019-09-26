import click
import numpy as np
import pandas as pd


@click.command()
@click.option('-i', '--input', type=click.Path(), required=True, multiple=True)
def main(input):
    input_to_df = {p: pd.read_csv(p) for p in input}
    mean = pd.DataFrame(
        {
            p1: [np.nan if p1 == p2 else (input_to_df[p1]['sirna'] == input_to_df[p2]['sirna']).mean() for p2 in input]
            for p1 in input
        },
        index=input)
    print(mean)

    eq = pd.DataFrame(
        {
            p1: [np.nan if p1 == p2 else (input_to_df[p1]['sirna'] != input_to_df[p2]['sirna']).sum() for p2 in input]
            for p1 in input
        },
        index=input)
    print(eq)


if __name__ == '__main__':
    main()
