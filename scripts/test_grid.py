

def main(args): 
    predictor = args.predictor
    corrector = args.corrector
    snr = args.snr

    print(f"{predictor} predictor steps, {corrector} corrector steps, snr ={snr}")


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--predictor")
    parser.add_argument("--corrector")
    parser.add_argument("--snr")
    args = parser.parse_args()
    main(args)