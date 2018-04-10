# -*- coding: utf-8 -*-

from csv import DictReader


class FNC_Data():

    # Define a class for holding the Fake News Challenge data

    # Self.stances contains all the headline and boby pairs (referenced using ids)
    # Self.headlines contains a unique set of headlines referenced by a headline_id
    # Self.bodies contains a unique set of bodies referenced by an body_id 

    def __init__(self, data_stances, data_bodies, data_path="fnc-1_data"):
        self.path = data_path

        self.stances = self.read(data_stances)
        self.headlines = {}

        # Process the stances data
        for stance in self.stances:
            if stance['Headline'] not in self.headlines.values():
                # Give each headline a unique id
                headline_id = len(self.headlines)
                self.headlines[headline_id] = stance['Headline']
                stance['Headline'] = headline_id
            else:
                # Get the headline_id
                temp = dict(zip(self.headlines.values(),self.headlines.keys()))
                headline_id = temp[stance['Headline']]
                stance['Headline'] = headline_id
            # Convert the Body id to integer
            stance['Body ID'] = int(stance['Body ID'])

        bodies = self.read(data_bodies)
        self.bodies = {}

        # Process the bodies
        for body in bodies:
            # Convert the Body id to integer
            self.bodies[int(body['Body ID'])] = body['articleBody']

        print ("Total bodies: " + str(len(self.bodies)))
        print ("Total headlines: " + str(len(self.headlines)))
        print ("Total stances: " + str(len(self.stances)))

    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding="utf-8") as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

