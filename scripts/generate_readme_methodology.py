import csv
import math
import pandas as pd
import copy

meta_methodology = {
    "Static Graph Construction": ["Dependency", "Constituency", "AMR", "IE", "Coreference", "Discourse", "Knowledge",
    "Social", "Topic", "Similarity", "Co-occurrence", "App-driven"],
    "Dynamic Graph Construction": ["Node Embedding Based", "Node Embedding Based Refined",
                                   "Node & Edge Embedding Based", "Node & Edge Embedding Based Refined"],
    "Graph Representation Learning": ["GNN for Undirected Graphs", "GNN for Directed Graphs",
                                      "GNN for Heterogeneous Graphs"],
    "GNN Based Encoder-Decoder": ["Graph2Seq", "Graph2Tree", "Graph2Graph"]
}

meth2class = {
   app: cls for cls, apps in meta_methodology.items() for app in apps
}

venue2year = {

}


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
            self.methodology = list(map(lambda x: x.strip(), info[4].split(",")))
            self.application = list(map(lambda x: x.strip(), info[5].split(",")))
            for meth in self.methodology:
                if meth not in meth2class:
                    print("The methodology: {} is not defined in paper information: {}".format(meth, line_info))
                    raise RuntimeError()
            self.owner = info[6]

    def __str__(self):
        return "title = {}, link = {}, bibtex = {}, venue = {}, methodology = {}, application = {}, owner = {}".format(
            self.title, self.link, self.bibtex, self.venue, self.methodology, self.application, self.owner
        )


class Container:
    class Category:
        def __init__(self, is_leaf: bool =False, children: list = None, type: str = None, value: str = "",
                     visible=True):
            self.is_leaf = is_leaf
            self.type = type
            assert self.type in ["application_super", "application",
                                 "methodology_super", "methodology",
                                 "venue_super", "venue", "root"]
            self.children = children
            self.value = value
            self.cnt = 0
            self.visible = visible

        def __eq__(self, other):
            return self.value == other

        def __str__(self):
            output = ""
            if self.type != "root" and self.cnt == 0:
                return output
            if self.type == "methodology_super":
                output += "- ## [" + self.value + "](#content)\n"
            elif self.type == "methodology":
                output += "\t - ### [" + self.value + "](#content)\n"
            elif self.type == "venue_super":
                output += "\t\t* #### Year: " + self.value + "\n"

            for node in self.children:
                if isinstance(node, Paper):
                    output += "\t\t\t[[{}] {}]({}) \n\n ".format(node.venue, node.title.strip(), node.link.strip())
                else:
                    output += str(node)
            return output

    def __init__(self, introduction_filename: str, papers: list):
        self.introduction_filename = introduction_filename
        self.papers = papers
        self.readme_content = ""
        self.saved_filename = "README_methodology.md"

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

        meta_meths = set(meth2class.keys())
        extra_meths = set(self.methodology) - meta_meths
        assert extra_meths == set()

        self._build_tree()
        self._scan_papers()

    def _scan_papers(self):
        for paper in self.papers:
            for meth in paper.methodology:
                venue = paper.venue
                success = self._insert_paper_in_tree(meth, venue, paper)
                if not success:
                    print("Paper not inserted: {}".format(paper))
                    exit(0)

        tree_cnt = 0
        for meth_super in self.tree.children:
            meth_super_cnt = 0
            for meth_node in meth_super.children:
                meth_node_cnt = 0
                for venue_super in meth_node.children:
                    venue_super_cnt = 0
                    for venue_node in venue_super.children:
                        venue_node.cnt = len(venue_node.children)
                        venue_super_cnt += venue_node.cnt
                    venue_super.cnt = venue_super_cnt
                    meth_node_cnt += venue_super_cnt
                meth_node.cnt = meth_node_cnt
                meth_super_cnt += meth_node.cnt
            meth_super.cnt = meth_super_cnt
            tree_cnt += meth_super.cnt
        self.tree.cnt = tree_cnt

    def _insert_paper_in_tree(self, meth, venue, paper):
        if meth not in meth2class:
            return False

        for meth_super in self.tree.children:
            if meth_super != meth2class[meth]:
                continue
            for meth_node in meth_super.children:
                if meth != meth_node:
                    continue
                for venue_super in meth_node.children:
                    if venue_super != venue2year[venue]:
                        continue
                    for venue_node in venue_super.children:
                        if venue != venue_node:
                            continue
                        venue_node.children.append(copy.deepcopy(paper))
                        return True
        return False

    def _build_tree(self):
        """
            application_cls
                application
                    venue
        :return:
        """
        venue_super_collect = []
        for year in ["2021", "2020", "2019", "2018", "2017"]:
            venue_super_children = []
            for venue in self.venue:
                venue_year = 2000 + int(venue.split("-")[1])
                if venue_year == int(year):
                    venue_node = self.Category(is_leaf=True, children=[], type="venue", value=venue, visible=False)
                    venue_super_children.append(venue_node)
                    venue2year[venue] = year
            venue_super_node = self.Category(is_leaf=False, children=venue_super_children, type="venue_super", value=year)
            venue_super_collect.append(venue_super_node)

        root_child = []

        for meth_super, meths in meta_methodology.items():
            meth_super_node = self.Category(is_leaf=False, children=[], type="methodology_super", value=meth_super)
            meth_super_children = []
            for meth in meths:
                meth_node = self.Category(is_leaf=False, children=copy.deepcopy(venue_super_collect), type="methodology", value=meth)
                meth_super_children.append(meth_node)
            meth_super_node.children = meth_super_children
            root_child.append(meth_super_node)

        self.tree = self.Category(is_leaf=False, children=root_child, type="root", value="")

    def generate_outline_table(self):

        table_content = ""
        for app_super_cnt, app_super in enumerate(self.tree.children):
            node_id = app_super.value.lower().replace("/", "").replace(" ", "-")
            app_super_content = '''<tr><td colspan="2"><a href="#{}">{}. {}: ({})</a></td></tr> \n'''.format(node_id,
                                            app_super_cnt + 1, app_super.value, app_super.cnt)
            app_node_content = ""
            app_cnt = 0
            for app in app_super.children:
                if app.cnt == 0:
                    continue
                node_app_id = app.value.lower().replace("/", "").replace(" ", "-")
                app_content = '''<td>&emsp;<a href="#{}">{}.{} {}: {}</a></td>'''.format(
                    node_app_id, app_super_cnt + 1, app_cnt + 1, app.value, app.cnt
                )
                app_node_content += app_content
                if (app_cnt + 1) % 2 == 0:
                    app_node_content = "\n<tr>\n{}\n</tr>\n".format(app_node_content)
                    app_super_content += app_node_content
                    app_node_content = ""
                app_cnt += 1

            if app_node_content != "":
                app_node_content = "\n<tr>\n{}\n</tr>\n".format(app_node_content)
                app_super_content += app_node_content

            table_content += app_super_content
        output = "## [Content](#content)\n\n" \
                 "<table>\n" \
                 "{}" \
                 "</table>\n".format(table_content)
        return output

    def save(self):
        output_str = self.readme_content
        output_str += "\n"
        outline = self.generate_outline_table()
        output_str += outline + "\n"
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
        try:
            paper = Paper(line)
            ret.append(paper)
        except Exception as e:
            print(e)
            print("Please fix annotation error")
            exit(0)
    return ret


if __name__ == "__main__":
    papers = read_from_xlsx("data/Survey_papers.xlsx")
    container = Container(introduction_filename="data/introduction_readme.md", papers=papers)
    container.build()
    container.save()
