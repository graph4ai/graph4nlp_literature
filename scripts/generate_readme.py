import csv
import math
import pandas as pd
import copy


class Paper:
    title = ""
    link = ""
    bibtex = ""
    venue = ""
    methodology = ""
    application = ""
    owner = ""

    def __init__(self, line_info=None):
        if line_info is not None:
            assert len(line_info) >= 7, print("Error exists in: {}, expect at least 7 attributes.")
            info = line_info[:7]
            self.title = info[0]
            self.link = info[1]
            self.bibtex = info[2]
            self.venue = info[3]
            print(self.venue, "&&&&&&&", type(self.venue))

            if isinstance(self.venue, float):
                print(line_info, "^^^^^^")

            try:
                self.methodology = list(map(lambda x: x.strip(), info[4].split(",")))
            except:
                print(info[4], "++++++")
                print(line_info)
                exit(0)
            try:
                self.application = list(map(lambda x: x.strip(), info[5].split(",")))
            except:
                print(info[5], "++++++")
                print(line_info)
                exit(0)
            self.owner = info[6]

    def __str__(self):
        return "title = {}, link = {}, bibtex = {}, venue = {}, methodology = {}, application = {}, owner = {}".format(
            self.title, self.link, self.bibtex, self.venue, self.methodology, self.application, self.owner
        )


class Container:
    class Category:
        def __init__(self, is_leaf: bool =False, children: list = None, type: str = None, value: str = ""):
            self.is_leaf = is_leaf
            self.type = type
            assert self.type in ["application", "methodology", "venue", "root"]
            self.children = children
            self.value = value
            self.cnt = 0

        def __eq__(self, other):
            return self.value == other

        def __str__(self):
            output = ""
            if self.type != "root" and self.cnt == 0:
                return output
            if self.type == "application":
                output += "- ## " + self.value + "\n"
            elif self.type == "methodology":
                output += "\t- ### " + self.value + "\n"
            elif self.type == "venue":
                output += "\t\t* #### " + self.value + "\n"
            else:
                output += "\n"
            for node in self.children:
                if isinstance(node, Paper):
                    output += "\t\t\t[{}]({}) \n\n ".format(node.title, node.link)
                else:
                    output += str(node)
            return output

    def __init__(self, introduction_filename: str, papers: list):
        self.introduction_filename = introduction_filename
        self.papers = papers
        self.readme_content = ""
        self.saved_filename = "README.md"

    def extract_unique_attrs(self, values):
        attrs = []
        for value in values:
            attrs.extend(value)
        return list(set(attrs))

    def build(self):
        # introductions
        self.readme_content = ""
        with open(self.introduction_filename, "r") as f:
            header = f.read()
        self.readme_content += header

        # papers
        self.applications = self.extract_unique_attrs(values=list(map(lambda x: x.application, self.papers)))

        self.methodology = self.extract_unique_attrs(values=list(map(lambda x: x.methodology, self.papers)))

        self.venue = self.extract_unique_attrs(values=list(map(lambda x: [x.venue], self.papers)))

        print("*********** Statistics *************")
        print("Application Information Summary:")
        print("There are {} applications as following: ".format(len(self.applications)))
        print(self.applications)
        print("Methodology Information Summary:")
        print("There are {} methodologies as following: ".format(len(self.methodology)))
        print(self.methodology)
        print("Venue Information Summary:")
        print("There are {} venues as following: ".format(len(self.venue)))
        print(self.venue)

        self._build_tree()
        self._scan_papers()

    def _scan_papers(self):
        for paper in self.papers:
            for app in paper.application:
                for meth in paper.methodology:
                    venue = paper.venue
                    success = self._insert_paper_in_tree(app, meth, venue, paper)
                    if not success:
                        print("Paper not inserted: {}".format(paper))
                        exit(0)

        for app_node in self.tree.children:
            app_node_cnt = 0
            for meth_node in app_node.children:
                meth_node_cnt = 0
                for venue_node in meth_node.children:
                    venue_node.cnt = len(venue_node.children)
                    meth_node_cnt += venue_node.cnt
                meth_node.cnt = meth_node_cnt
                app_node_cnt += meth_node.cnt
            app_node.cnt = app_node_cnt

    def _insert_paper_in_tree(self, app, meth, venue, paper):
        for app_node in self.tree.children:
            if app != app_node:
                continue
            for meth_node in app_node.children:
                if meth != meth_node:
                    continue
                for venue_node in meth_node.children:
                    if venue != venue_node:
                        continue
                    venue_node.children.append(copy.deepcopy(paper))
                    return True
        return False

    def _build_tree(self):
        """
            application
                methodology
                    venue
        :return:
        """
        root_child = []
        for app in self.applications:
            app_child = []
            for methodology in self.methodology:
                methodology_child = []
                for venue in self.venue:
                    venue_node = self.Category(is_leaf=True, children=[], type="venue", value=venue)
                    methodology_child.append(venue_node)
                methodology_node = self.Category(is_leaf=False, children=methodology_child, type="methodology", value=methodology)
                app_child.append(methodology_node)
            app_node = self.Category(is_leaf=False, children=app_child, type="application", value=app)
            root_child.append(app_node)
        self.tree = self.Category(is_leaf=False, children=root_child, type="root", value="")

    def save(self):
        output_str = self.readme_content
        output_str += "\n"
        output_str += str(self.tree)
        with open(self.saved_filename, "w") as f:
            f.write(output_str)



        print("Readme file: {} saved successfully!".format(self.saved_filename))


def read_from_xlsx(filename):
    data = pd.read_excel(filename, sheet_name="papers")
    ret = []
    for line in data.values:
        if isinstance(line[0], float) and math.isnan(line[0]):
            continue
        paper = Paper(line)
        ret.append(paper)
    return ret


if __name__ == "__main__":
    papers = read_from_xlsx("data/Survey_papers.xlsx")
    container = Container(introduction_filename="data/introduction_readme.md", papers=papers)
    container.build()
    container.save()
