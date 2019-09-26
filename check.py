import click
import pandas as pd


@click.command()
@click.option('--a', type=click.Path(), required=True)
@click.option('--b', type=click.Path(), required=True)
def main(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)

    eq = a['sirna'] == b['sirna']
    print(eq.sum(), len(eq))
    print(eq.mean())


if __name__ == '__main__':
    main()
