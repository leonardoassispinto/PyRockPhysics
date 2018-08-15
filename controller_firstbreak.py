import argparse
import firstbreak

if __name__ == '__main__':
            
    # Parsing command line input
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face_to_face", type=str, action='append', required=True, help="Face-to-Face file")
    ap.add_argument("-p", "--sample_file", type=str, action='append', required=True, help="Sample file")
    ap.add_argument("-l", "--length", type=float, required=True, help="Core length")
    ap.add_argument("-s", "--sample_type", required=True, help="Measured sample type")
    
    args = vars(ap.parse_args())

    ftf_file = args["face_to_face"]
    s_file = args["sample_file"]
    L = args["length"]
    sample_type = args["sample_type"]
    firstbreak.first_break(ftf_file, s_file, L, sample_type)