import uuid
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Replace hits evnetID from local increasing sequence to a UUID.')
parser.add_argument('path', type=str, default='./hits.csv', help='Path of imput hits csv file.')

args = parser.parse_args()


def gen_uuid4():
    yield str(uuid.uuid4())


hits = pd.read_csv(args.path)
uuid_mapping = {eventID: next(gen_uuid4()) for eventID in hits['eventID'].unique()}
hits['eventID'] = pd.Series([uuid_mapping[eventID] for eventID in hits['eventID']])
hits.to_csv(args.path, index=False)
hits[hits['PDGEncoding']==22].to_csv('gamma_hits.csv', index=False)

