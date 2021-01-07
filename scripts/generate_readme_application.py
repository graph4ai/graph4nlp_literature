import csv
import math
import pandas as pd
import copy

meta_application = {
    "Knowledge Graph/Knowledge Base": ["Knowledge Graph Embedding", "Knowledge Base Completion",
                                       "Knowledge Graph Alignment"],
    "Information Extraction": ["Named-entity Recognition", "Relation Extraction", "Event Detection"],
    "Sequence Labeling": ["POS Tagging", "Semantic Role Labeling"],
    "Natural Language Generation": ["Machine Translation", "Summarization", "Code Summarization",
                                    "Question Generation", "AMR2Text", "SQL2Text"],
    "Question Answering": ["Machine Reading Comprehension", "Knowledge Base Question Answering",
                           "Open-domain Question Answering", "Community Question Answering"],
    "Parsing": ["Dependency Parsing", "AMR Parsing", "Semantic Parsing", "Constituency parsing"],
    "Reasoning": ["Natural Language Inference", "Math Word Problem", "Commonsense Reasoning"],
    "Dialog Systems": ["Dialogue State Tracking", "Dialogue Generation", "Next Utterance Prediction"],
    "Text Classification": ["Text Classification"],
    "Text Matching": ["Text Matching"],
    "Topic Modeling": ["Topic Modeling"],
    "Sentiment Analysis": ["Sentiment Analysis"]
}

app2class = {
   app: cls for cls, apps in meta_application.items() for app in apps
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
            for app in self.application:
                if app not in app2class:
                    print("The application: {} is not defined in paper information: {}".format(app, line_info))
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
            assert self.type in ["application_super", "application", "methodology",
                                 "venue_super", "venue", "root"]
            self.children = children
            self.value = value
            self.cnt = 0
            self.visible = visible

        def __eq__(self, other):
            return self.value == other

        def convert(self):
            assert self.type == "root"
            output = "- ## Knowledge Graph/Knowledge Base\n" \
                     "{knowledge} \n" \
                     "- ## Information Extraction\n" \
                     "{ie} \n" \
                     "- ## Sequence Labeling\n" \
                     "{label} \n" \
                     "- ## Natural Language Generation\n" \
                     "{nlg} \n" \
                     "- ## Question Answering\n" \
                     "{Parsing} \n" \
                     "- ## Reasoning\n" \
                     "{reason}\n" \
                     "- ## Dialog Systems\n" \
                     "{dialog}\n" \
                     "- ## Text Classification\n" \
                     "{textcls}\n" \
                     "- ## Text Matching\n" \
                     "{textmatch}\n" \
                     "- ## Topic Modeling\n" \
                     "{topic}\n" \
                     "- ## Sentiment Analysis\n" \
                     "{sa}\n"

        def __str__(self):
            output = ""
            if self.type != "root" and self.cnt == 0:
                return output
            if self.type == "application_super":
                output += "- ## [" + self.value + "](#content)\n"
            elif self.type == "application":
                output += "\t - ### [" + self.value + "](#content)\n"
            elif self.type == "venue_super":
                output += "\t\t* #### Year: " + self.value + "\n"

            for node in self.children:
                if isinstance(node, Paper):
                    output += "\t\t\t[({}) {}]({}) \n\n ".format(node.venue, node.title.strip(), node.link.strip())
                else:
                    output += str(node)
            return output

    def __init__(self, introduction_filename: str, papers: list):
        self.introduction_filename = introduction_filename
        self.papers = papers
        self.readme_content = ""
        self.saved_filename = "README_application.md"

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

        meta_apps = set(app2class.keys())
        extra_apps = set(self.applications) - meta_apps
        assert extra_apps == set()

        self._build_tree()
        self._scan_papers()

    def _scan_papers(self):
        for paper in self.papers:
            for app in paper.application:
                venue = paper.venue
                success = self._insert_paper_in_tree(app, venue, paper)
                if not success:
                    print("Paper not inserted: {}".format(paper))
                    exit(0)

        tree_cnt = 0
        for app_super in self.tree.children:
            app_super_cnt = 0
            for app_node in app_super.children:
                app_node_cnt = 0
                for venue_super in app_node.children:
                    venue_super_cnt = 0
                    for venue_node in venue_super.children:
                        venue_node.cnt = len(venue_node.children)
                        venue_super_cnt += venue_node.cnt
                    venue_super.cnt = venue_super_cnt
                    app_node_cnt += venue_super_cnt
                app_node.cnt = app_node_cnt
                app_super_cnt += app_node.cnt
            app_super.cnt = app_super_cnt
            tree_cnt += app_super.cnt
        self.tree.cnt = tree_cnt

    def _insert_paper_in_tree(self, app, venue, paper):
        if app not in app2class:
            return False

        for app_super in self.tree.children:
            if app_super != app2class[app]:
                continue
            for app_node in app_super.children:
                if app != app_node:
                    continue
                for venue_super in app_node.children:
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

        for app_super, apps in meta_application.items():
            app_super_node = self.Category(is_leaf=False, children=[], type="application_super", value=app_super)
            app_super_children = []
            for app in apps:
                app_node = self.Category(is_leaf=False, children=copy.deepcopy(venue_super_collect), type="application", value=app)
                app_super_children.append(app_node)
            app_super_node.children = app_super_children
            root_child.append(app_super_node)

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
        output = "## [Content: ({})](#content)\n\n" \
                 "<table>\n" \
                 "{}" \
                 "</table>\n".format(self.tree.cnt, table_content)
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
