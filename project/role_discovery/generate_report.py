import argparse
from .utils.analysis import generate_comparison_summary

def main():
    parser = argparse.ArgumentParser(description="Generate summary reports for Role Discovery experiments.")
    parser.add_argument('--dataset', type=str, nargs='+', default=['Cora', 'Actor', 'CLUSTER'],
                        help="A list of datasets to include in the summary report.")
    args = parser.parse_args()

    print(f"Generating summary report for datasets: {', '.join(args.dataset)}")
    generate_comparison_summary(args.dataset)
    print("\nReport generation finished.")

if __name__ == '__main__':
    main()