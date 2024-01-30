import json
import datetime


class PromptGenerator:
    def __init__(self, dict_id2ont, dict_hier_id):
        self.dict_id2ont = dict_id2ont
        self.dict_hier_id = dict_hier_id


    def get_prompt_1_relation(self, article):

        first_ont_list = []
        for i in range(1, 21):
            curr_id = str(i) + "0" if i > 9 else "0" + str(i) + "0"
            first_ont_list.append(self.dict_id2ont[curr_id]["choice"])
        rules = [
            "1. Extract each event in format: event actor 1; event relation; event actor 2.",
            "2. Only choose event relation from this relation candidate list: " + ', '.join(first_ont_list) + '.',
            "3. Event actors are usually political actors, countries or international organizations.",
            "4. Only extract events that have happened or is happening, and not extract future events."
        ]
        prompt_rules = f'You are an assistant to perform structured event extraction from news articles with following rules:\n' + '\n'.join(rules)

        prompt_example = "For example, given the example article:\n" + \
            "Egypt committed to boosting economic cooperation with Lebanon\n" +\
            "(MENAFN- Daily News Egypt) Egypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.\n" +\
            "\nList all events by rules, the extraction result of the example is:\n" +\
            "Egypt; Express intent to cooperate; Lebanon | Egypt president Abdel Fattah Al-Sisi; Consult or meet; Lebanese parliamentary speaker Nabih Berri | Lebanese parliamentary speaker Nabih Berri; Consult or meet; Egypt president Abdel Fattah Al-Sisi\n"

        text_article = article['Title'] + '\n' + ('\n'.join(article['Text'][:3]) if len(article['Text']) >=3 else '\n'.join(article['Text']))
        raw_tokens = text_article.split(' ')
        if len(raw_tokens) > 512:
            text_article = ' '.join(raw_tokens[:512])

        prompt_instruction = 'Now, given the query artcile:\n' + text_article + '\n\nList all events by rules, the extraction result of the query article is:'

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


    def get_prompt_2_relation(self, article, row):
        s, r_id, r_choice, o, md5 = row
        level2_ids = list(self.dict_hier_id[r_id].keys())
        level2_choices = [self.dict_id2ont[idx]['choice'] for idx in level2_ids]

        rules = [
            "1. A original structured event is given in format: event actor 1 (subject); event relation; event actor 2 (object).",
            "2. All sub-relation candidates of the original event relation are also given.",
            "3. A news article where the original structured event is extracted from is also given.",
            "4. Based on the article, only choose one best sub-relation from the given candidate list that best matches the article, subject and object. The answer can be 'Not specified'."
        ]
        prompt_rules = f'You are an assistant to perform structured event extraction from news articles with following rules:\n' + '\n'.join(rules)

        prompt_example = "For example, given the structured event: Egypt; Express intent to cooperate; Lebanon.\n" + \
            "And given the sub-relation candidate list: Not specified; Express intent to engage in material cooperation; Express intent to engage in diplomatic cooperation; Express intent to yield or concede; Express intent to mediate.\n" + \
            "And given the news article:\n" + \
            "(MENAFN- Daily News Egypt)\nEgypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.\n" + \
            "\nChoose by rules, the sub-relation in this example is:\nExpress intent to engage in material cooperation.\n"

        text_article = article['Title'] + '\n' + ('\n'.join(article['Text'][:3]) if len(article['Text']) >= 3 else '\n'.join(article['Text']))
        raw_tokens = text_article.split(' ')
        if len(raw_tokens) > 512:
            text_article = ' '.join(raw_tokens[:512])

        prompt_instruction = 'Now, given the structured event: ' + '; '.join([s, r_choice, o]) + '\n' + \
            "And given the sub-relation candidate list: Not specified; " + '; '.join(level2_choices) + '.\n' +\
            "And given the news article:\n" + text_article + '\n' + \
            "\nChoose by rules, the sub-relation in this example is:"

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])

    def get_prompt_3_relation(self, article, row):
        level1_rowid, s, r_id, r_choice, o, md5 = row
        level3_ids = self.dict_hier_id[r_id]
        level3_choices = [self.dict_id2ont[idx]['choice'] for idx in level3_ids]

        rules = [
            "1. A original structured event is given in format: event actor 1 (subject); event relation; event actor 2 (object).",
            "2. All sub-relation candidates of the original event relation are also given.",
            "3. A news article where the original structured event is extracted from is also given.",
            "4. Based on the article, only choose one best sub-relation from the given candidate list that best matches the article, subject and object. The answer can be 'Not specified'."
        ]
        prompt_rules = f'You are an assistant to perform structured event extraction from news articles with following rules:\n' + '\n'.join(rules)

        prompt_example = "For example, given the structured event: Egypt; Express intent to material cooperate; Lebanon.\n" + \
            "And given the sub-relation candidate list: Not specified; Express intent to cooperate economically; Express intent to cooperate militarily; Express intent to cooperate on judicial matters; Express intent to cooperate on intelligence or information sharing.\n" + \
            "And given the news article:\n" + \
            "(MENAFN- Daily News Egypt)\nEgypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.\n" + \
            "\nChoose by rules, the sub-relation in this example is:\nExpress intent to cooperate economically.\n"

        text_article = article['Title'] + '\n' + ('\n'.join(article['Text'][:3]) if len(article['Text']) >= 3 else '\n'.join(article['Text']))
        raw_tokens = text_article.split(' ')
        if len(raw_tokens) > 512:
            text_article = ' '.join(raw_tokens[:512])

        prompt_instruction = 'Now, given the structured event: ' + '; '.join([s, r_choice, o]) + '\n' + \
            "And given the sub-relation candidate list: Not specified; " + '; '.join(level3_choices) + '.\n' +\
            "And given the news article:\n" + text_article + '\n' + \
            "\nChoose by rules, the sub-relation in this example is:"

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


    def get_date_str(self, date):
        date = str(date)
        year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
        date_obj = datetime.datetime(year, month, day)
        date_str = date_obj.strftime('%Y %B %d, %A')  # eg. '2023 April 06, Thursday'

        return date_str
