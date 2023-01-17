import sys
import ipdb

INPUT_STYLE_SET = ['event_type', 'event_type_sent', 'keywords', 'triggers', 'template']
OUTPUT_STYLE_SET = ['trigger:sentence', 'argument:sentence']


class eve_template_generator():
    def __init__(self, passage, triggers, roles, input_style, output_style, vocab, instance_base=False):
        """
        generate strctured information for events
        
        args:
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            input_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
            instance_base(Bool): if instance_base, we generate only one pair (use for trigger generation), else, we generate trigger_base (use for argument generation)
        """
        self.raw_passage = passage
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style
        self.vocab = vocab
        self.event_templates = []
        if instance_base:
            for e_type in self.vocab['event_type_itos']:
                theclass = getattr(sys.modules[__name__], e_type.replace(':', '_').replace('-', '_'), False)
                if theclass:
                    self.event_templates.append(theclass(self.input_style, self.output_style, passage, e_type, self.events))
                else:
                    print(e_type)

        else:
            for event in self.events:
                theclass = getattr(sys.modules[__name__], event['event type'].replace(':', '_').replace('-', '_'), False)
                assert theclass
                self.event_templates.append(theclass(self.input_style, self.output_style, event['tokens'], event['event type'], event))
        self.data = [x.generate_pair(x.trigger_text) for x in self.event_templates]
        self.data = [x for x in self.data if x]

    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = ' '.join(passage[trigger[0]:trigger[1]])
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': ' '.join(passage[argument[1][0]:argument[1][1]]),
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ' '.join(passage),
                'tokens': passage
            })
        return event_structures

class event_template():
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.passage = ' '.join(passage)
        self.tokens = passage
        self.event_type = event_type
        if gold_event is not None:
            self.gold_event = gold_event
            if isinstance(gold_event, list):
                # instance base
                self.trigger_text = " and ".join([x['trigger text'] for x in gold_event if x['event type']==event_type])
                self.trigger_span = [x['trigger span'] for x in gold_event if x['event type']==event_type]
                self.arguments = [x['arguments'] for x in gold_event if x['event type']==event_type]
            else:
                # trigger base
                self.trigger_text = gold_event['trigger text']
                self.trigger_span = [gold_event['trigger span']]
                self.arguments = [gold_event['arguments']]         
        else:
            self.gold_event = None
        
    @classmethod
    def get_keywords(self):
        pass

    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair
        """
        input_str = self.generate_input_str(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)

    def generate_input_str(self, query_trigger):
        return None

    def generate_output_str(self, query_trigger):
        return (None, False)

    def decode(self, prediction):
        pass

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        # categorize prediction
        pred_trigger = []
        pred_argument = []
        for pred in predict_output:
            if pred[1] == self.event_type:
                pred_trigger.append(pred)
            else:
                pred_argument.append(pred)
        # trigger score
        gold_tri_num = len(self.trigger_span)
        pred_tris = []
        for pred in pred_trigger:
            pred_span = self.predstr2span(pred[0])
            if pred_span[0] > -1:
                pred_tris.append((pred_span[0], pred_span[1], pred[1]))
        pred_tri_num = len(pred_tris)
        match_tri = 0
        for pred in pred_tris:
            id_flag = False
            for gold_span in self.trigger_span:
                if gold_span[0] == pred[0] and gold_span[1] == pred[1]:
                    id_flag = True
            match_tri += int(id_flag)

        # argument score
        converted_gold = self.get_converted_gold()
        gold_arg_num = len(converted_gold)
        pred_arg = []
        for pred in pred_argument:
            # find corresponding trigger
            pred_span = None
            if isinstance(self.gold_event, list):
                # end2end case
                try:
                    # we need this ``try'' because we cannot gurantee the model will be bug-free on the matching
                    cor_tri = pred_trigger[pred[2]['cor tri cnt']]
                    cor_tri_span = self.predstr2span(cor_tri[0])[0]
                    if cor_tri_span > -1:
                        pred_span = self.predstr2span(pred[0], cor_tri_span)
                    else:
                        continue
                except Exception as e:
                    print(e)
            else:
                # argument only case
                pred_span = self.predstr2span(pred[0], self.trigger_span[0][0])
            if (pred_span is not None) and (pred_span[0] > -1):
                pred_arg.append((pred_span[0], pred_span[1], pred[1]))
        pred_arg = list(set(pred_arg))
        pred_arg_num = len(pred_arg)
        
        target = converted_gold
        match_id = 0
        match_type = 0
        for pred in pred_arg:
            id_flag = False
            id_type = False
            for gold in target:
                if gold[0]==pred[0] and gold[1]==pred[1]:
                    id_flag = True
                    if gold[2] == pred[2]:
                        id_type = True
                        break
            match_id += int(id_flag)
            match_type += int(id_type)
        return {
            'gold_tri_num': gold_tri_num, 
            'pred_tri_num': pred_tri_num,
            'match_tri_num': match_tri,
            'gold_arg_num': gold_arg_num,
            'pred_arg_num': pred_arg_num,
            'match_arg_id': match_id,
            'match_arg_cls': match_type
        }
    
    def get_converted_gold(self):
        converted_gold = []
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold.append((arg['argument span'][0], arg['argument span'][1], arg_type))
        return list(set(converted_gold))
    
    def predstr2span(self, pred_str, trigger_idx=None):
        sub_words = [_.strip() for _ in pred_str.strip().lower().split()]
        candidates=[]
        for i in range(len(self.tokens)):
            j = 0
            while j < len(sub_words) and i+j < len(self.tokens):
                if self.tokens[i+j].lower() == sub_words[j]:
                    j += 1
                else:
                    break
            if j == len(sub_words):
                candidates.append((i, i+len(sub_words)))
        if len(candidates) < 1:
            return -1, -1
        else:
            if trigger_idx is not None:
                return sorted(candidates, key=lambda x: abs(trigger_idx-x[0]))[0]
            else:
                return candidates[0]



class Hostile_encounter(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['battle', 'fight', 'contest']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a hostile encounter between opposing forces')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Arrest(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bust', 'book', 'imprison']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a person getting arrested or a person being sent to jail.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Expansion(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['enlarge', 'grow', 'inflate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an expansion of size')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Coming_to_believe(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['conclude', 'determine', 'surmise']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to coming to believe something after reasoning')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Attack(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['assault', 'strike', 'hit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to conflict and some violent act.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Education_teaching(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['educate', 'lecture', 'train']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to teaching')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Using(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['use', 'apply', 'employ']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent manipulating something to achieve a purpose')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Reporting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['inform', 'report', 'tell']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent reporting information ')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Change_event_time(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['advance', 'delay', 'move up']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to changing the time or duration of an event')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Recovering(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['recover', 'refresh', 'restore']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to returning to an earlier state of strength or vigor')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Process_start(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['begin', 'commence', 'start']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something beginning')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cause_to_be_included(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['add', 'include']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something causing a member to be included in a group')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Coming_to_be(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['appear', 'emerge', 'take shape']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something coming into existence')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Traveling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['travel', 'commute', 'tour']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something going on a journey')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Presence(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['present', 'presence', 'absence']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the existence of an entity at a particular location or time')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Deciding(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['decide', 'determine', 'rule out']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to deciding between options')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Receiving(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['accept', 'receive', 'obtain']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a recipient receiving something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Body_movement(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bend', 'kneel', 'wink']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent performing an action using some part of their body.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Change(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['convert', 'modify', 'ship']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a change')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Change_of_leadership(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['elect', 'oust', 'take over']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a change in leadership')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Placing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['place', 'find', 'arrange']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a change in something\'s physical location')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cause_change_of_strength(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['strengthen', 'weaken', 'fortify']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a change in strength')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Commitment(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['consent', 'pledge', 'vow']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a commitment or promise')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Come_together(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['assemble', 'convene', 'gather']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a group of individuals coming together')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Warning(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['alert', 'warn', 'forwarn']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a message that describes an undesirable situation')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Military_operation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['campaign', 'operate', 'operation']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a military operation')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Response(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['meet', 'react', 'respond']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a response to an occurance')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Telling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['tell', 'notify', 'advise']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a speaker addressing someone with a message')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Temporary_stay(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['guest', 'overnight', 'lodge']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to a temporary stay somehwere')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Adducing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['cite', 'name', 'point']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to adducing or citing as evidence')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Agree_or_refuse_to_act(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['agree', 'decline', 'refusal']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to agreeing to or refusing to engage with an action')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Aiming(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['aim', 'direct', 'target']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to aiming an instrument')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Action(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['nan']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an action')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Achieve(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['accomplish', 'bring about', 'effect']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent achieving something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Destroying(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['annihilate', 'destroy', 'dismantle']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent affecting something so negatively it no longer exists')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Change_sentiment(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['amuse', 'entertain', 'terrorize']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent causing a change in sentiment or feeling')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Damaging(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['damage', 'vandalize', 'deface']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent changing something else to an undesirable state')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Preventing_or_letting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['prevent', 'avoid', 'allow']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent keeping something from taking place or allowing it to happen')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Check(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['confirm', 'identify', 'certify']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an agent verifying knowledge')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Emergency(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['emergency', 'wildfire', 'fire']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an emergency')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Self_motion(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['advance', 'dash', 'march']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an entity moving under its own direction')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Supporting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bolster', 'buttress', 'support']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an entity supporting or aiding another ')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Catastrophe(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['disaster', 'crisis', 'accident']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an extremely undesirable occurance')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Incident(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['proved', 'happen', 'random']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to an incident or coincidence')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Arranging(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['array', 'deploy', 'format']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to arranging something in a configuration')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Award(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['award', 'deserve', 'merit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to awarding or deserving')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Rewards_and_punishments(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['discipline', 'penalty', 'reward']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to awards and punishments')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Becoming(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['become', 'turn', 'transform']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to becoming something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Besieging(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['besiege', 'encircle', 'invest']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to besieging a location, particularly in a military context')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Breathing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['breath', 'exhale', 'inhale']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to breathing')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Bringing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['carry', 'fetch', 'lug']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to bringing something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Building(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['assemble', 'make', 'erect']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to building something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Commerce_buy(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['buy', 'purchase', 'client']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to buying something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Carry_goods(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['carry', 'stock', 'distribute']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to carrying or distributing goods')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Causation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['because', 'cause', 'create']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to causing something to happen')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cause_to_make_progress(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['advance', 'improve', 'pefect']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to causing something to make progress')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Choosing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['opt', 'pick', 'select']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to choosing something out of a set of possibilties')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Collaboration(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['collude', 'cooperate', 'together']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to collaboration between entities')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Committing_crime(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['commit', 'crime', 'perpetrate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to commiting a crime')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Communication(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['convey', 'indicate', 'say']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to communicating information')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Competition(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['compete', 'game', 'play']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to competitions')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Confronting_problem(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['confrant', 'face', 'deal with']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to confronting a problem')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Connect(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['against', 'on', 'upon']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to connection or spatial contact between entities')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Expend_resource(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['deplete', 'expend', 'use up']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to consuming a resource')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Containing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['hold', 'house', 'store']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to containing or holding within physical boundaries')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Convincing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['dissuade', 'motivate', 'persuade']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to convincing or attempting to convince someone')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Creating(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['generate', 'produce', 'assemble']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to creating an entity')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Create_artwork(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['draw', 'paint', 'sculpt']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to creating artwork')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Criminal_investigation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['clue', 'investigate', 'probe']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to criminal investigations')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cure(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['heal', 'rehabilitate', 'treat']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to curing or healing something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Death(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['die', 'end', 'expire']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to death')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Defending(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['defend', 'hold', 'resist']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to defending or responding to an attack')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Departing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['depart', 'exit', 'leave']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to departing')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Earnings_and_losses(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['make', 'profit', 'result']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to earning and losing things')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Employment(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['employ', 'hire', 'work']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to employment')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Emptying(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['disarm', 'clean', 'strip']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to emptying containers and clearing areas of some substance or item')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Forming_relationships(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['befriend', 'marry', 'separate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to entities forming relationships')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Exchange(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['convert', 'change', 'switch']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to exchanging something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Research(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['investigate', 'study', 'test']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to experimentation or research')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Extradition(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['extradition', 'deport', 'expel']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to extradition')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Filling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['brush', 'cover', 'fill']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to filling containers and covering areas with something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class GetReady(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['prepare', 'ready', 'set']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to getting ready for something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Getting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['acquire', 'obtain', 'secure']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to getting something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Giving(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bequeath', 'give', 'pass']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to giving something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class GiveUp(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['abandon', 'forget', 'leave']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to giving up or abandoning')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Having_or_lacking_access(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['access', 'block', 'inaccessible']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to having or lacking access to something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Bearing_arms(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bear arms', 'pack', 'carry']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to having weapons')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Hindering(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['hold back', 'inhibit', 'interfere']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to hindering a process')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Imposing_obligation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['charge', 'obligate', 'pledge']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to imposing an obligation')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Ingestion(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['consume', 'eat', 'devour']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to ingestion')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Institutionalization(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['admit', 'commit', 'institutionalize']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to institutionalization or hospitalization')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cause_to_amalgamate(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['amalgamate', 'combine', 'merge']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to joining parts to form a whole')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Justifying(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['account', 'defend', 'rationalize']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to justifying an action or belief')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Kidnapping(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['abduct', 'nab', 'snatch']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to kidnapping')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Labeling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['brand', 'call', 'term']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to labeling')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Legal_rulings(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['adjudicate', 'judge', 'rule']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to legal rulings ')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Limiting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['limit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to limiting or limitations')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Statement(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['address', 'explain', 'say']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to making a statement of some information')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Manufacturing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['make', 'produce', 'fabricate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to manufacturing a product')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Motion(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['come', 'move', 'go']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to movement through space')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Motion_directional(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['descend', 'drop', 'rise']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to moving in a direction')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Participation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['engage', 'involved', 'party']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to participating in something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Patrolling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['patrol', 'police', 'observe']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to patrolling an area')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Commerce_pay(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['disburse', 'pay', 'shell out']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to paying for something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Practice(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['practice', 'mock', 'rehearse']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to practicing something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Preserving(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['cure', 'embalm', 'preserve']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to preserving something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Prison(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['brig', 'jail', 'prison']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to prison or penal institutions')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Assistance(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['aid', 'help', 'serve']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to providing assistance')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Expressing_publicly(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['articulate', 'express', 'vent']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to publicly expressing some information')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Publishing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['publish', 'releash', 'appear']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to publishing or officially releasing information')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Quarreling(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['fight', 'quibble', 'dispute']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to quarreling or arguing')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Ratification(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['ratify', 'ratification', 'treaty']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to ratification')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Recording(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['document', 'record', 'register']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to recording information')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Reforming_a_system(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['overhaul', 'reform', 'restructure']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to reforming a system')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Renting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['hire', 'lease', 'rent']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to renting something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Rescuing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['rescue', 'save']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to rescuing')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Resolve_problem(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['correct', 'handle', 'solve']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to resolving a problem')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Reveal_secret(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['disclose', 'reveal', 'slip']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to revealing a secret')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Revenge(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['avenge', 'retaliate', 'vengeful']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to revenge')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Risk(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['chance', 'dare', 'hazard']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to risk or risky actions')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Rite(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['baptize', 'pray', 'worship']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to rites or rituals')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Robbery(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['rob', 'steal', 'mug']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to robbery and theft')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Scouring(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['rummage', 'scour', 'sift']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to scouring or searching for something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Scrutiny(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['analyze', 'investigate', 'search']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to scrunity or inspection')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Commerce_sell(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['auction', 'retail', 'vend']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to selling something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Sign_agreement(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['accede', 'sign', 'signature']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to signing an agreement')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Social_event(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['celebrate', 'party', 'gala']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to social gatherings')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Perception_active(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['look', 'observe', 'feel']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to some agent directing their attention to something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Control(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['command', 'control', 'regulate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to some entity having control over another')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Know(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['realized', 'known', 'realise']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone adding knowledge to their model of the world.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Request(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['request', 'implore', 'demand']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone asking for something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Becoming_a_member(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['enroll', 'join', 'sign up']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone becoming a member of a group')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Bodily_harm(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['break', 'scrape', 'tear']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone experiencing a bodily injury')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Surrendering(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['give up', 'surrender', 'turn in']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone surrendering')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Suspicion(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['suspect', 'fear', 'think']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to someone suspecting someone else of something negative')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Arriving(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['arrive', 'approach', 'come']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something arriving at a location or state')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Being_in_operation(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['on', 'off', 'operate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something being in operation')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Process_end(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['conclude', 'end', 'final']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something ending')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Influence(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['affect', 'impact', 'inspire']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something influencing something else')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Killing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['kill', 'murder', 'slay']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to something that causes the death of a victim. ')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Submitting_documents(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['file', 'hand in', 'submit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to submitting documents')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Supply(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['equip', 'issue', 'provide']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to supplying something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Surrounding(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['around', 'rim', 'encompass']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to surrounding something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Terrorism(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bioterrorism', 'ecoterrorism', 'terror']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to terrorism')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Testing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['assessment', 'exam', 'test']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to testing or examination')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Change_tool(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['change', 'switch', 'transfer']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the change of a tool')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cause_change_of_position_on_a_scale(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['add', 'diminish', 'raise']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the change of position of an item from an initial value to an end value')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Judgment_communication(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['charge', 'cite', 'denounce']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the communication of a judgement')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Dispersal(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['dissolve', 'scatter', 'spread']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the dispersal of something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Hiding_objects(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['block', 'cover', 'mask']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the hiding of objects')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Conquering(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['conquer', 'invade', 'take']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the invasion or taking over of an entity')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Legality(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['illegal', 'legal', 'prohibited']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the legality of an action')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Hold(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['hold', 'pull', 'touch']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the manipulation of an entity by an agent')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Lighting(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['glint', 'shine', 'twinkle']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the movement of light')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Openness(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['closed', 'dark', 'open']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the openess or accessibility of something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Protest(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['argue', 'demonstrate', 'prove']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the presentation of content with support')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Releasing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['release', 'set free', 'let go']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the release of an entity.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Removing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['discard', 'eliminate', 'take']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the removal of something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Sending(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['send', 'deliver', 'dispatch']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the transfer of something ')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Vocalizations(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['cry', 'howl', 'squeak']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to the vocalizations of people or animals')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Theft(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['steal', 'shoplift', 'heist']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to theft')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Use_firearm(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['discharge', 'fire', 'shoot']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to using a firearm')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Violence(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['brutality', 'savagery', 'violence']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to violence')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Wearing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['bare', 'clothed', 'nude']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to wearing or dressing')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Writing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['draft', 'pen', 'type']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to writing or text creation')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Escaping(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['escape', 'evacuate', 'get out']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is relating to escaping from an undesirable place')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Cost(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['cheap', 'expense', 'pricy']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is relayed to the cost or price of something')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output


class Name_conferral(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['call', 'dub', 'name']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('{}')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('This event is related to conferring a name')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
        return output

