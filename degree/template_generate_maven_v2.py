
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

class Achieve(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['achieved', 'reached', 'achieve', 'success', 'attained', 'reach']

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
                    input_str += ' \n {}'.format('This is a Achieve event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['conducted', 'carried out', 'made', 'conduct', 'act', 'implemented']

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
                    input_str += ' \n {}'.format('This is a Action event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['referred', 'cited', 'citing', 'refers', 'refer', 'cite']

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
                    input_str += ' \n {}'.format('This event is named Adducing and its definition is "a Speaker mentions a specified Entity filling a role through a medium."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['refused', 'agreed', 'opposed', 'rejected', 'denied', 'agree']

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
                    input_str += ' \n {}'.format('This event is named Agree_or_refuse_to_act and its definition is "a Speaker agrees or refuses to engage in a proposed Action from an Interlocutor."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['targeted', 'aimed', 'target', 'focused', 'aim', 'targeting']

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
                    input_str += ' \n {}'.format('This event is named Aiming and its definition is "an Agent adjusts an Instrument to enable it to interact directly with a difficult to access Target_location."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['organized', 'planned', 'organised', 'ordered', 'deployed', 'plan']

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
                    input_str += ' \n {}'.format('This event is named Arranging and its definition is "An Agent puts a complex Theme into a particular Configuration."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['arrested', 'arrest', 'detained', 'imprisoned', 'captured', 'interned']

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
                    input_str += ' \n {}'.format('This event is named Arrest and its definition is "Authorities charge a Suspect who is under suspicion of having committed a crime the Charges."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['reached', 'entered', 'returned', 'reaching', 'landfall', 'arrived']

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
                    input_str += ' \n {}'.format('This event is named Arriving and its definition is "an object Theme moves in the direction of a Goal."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['help', 'helped', 'aid', 'served', 'assistance', 'aided']

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
                    input_str += ' \n {}'.format('This event is named Assistance and its definition is "a Helper benefits a Benefited_party by enabling the culmination of a Goal that the Benefited_party has."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['attack', 'attacked', 'bombing', 'invasion', 'struck', 'raid']

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
                    input_str += ' \n {}'.format('This event is named Attack and its definition is "an Assailant physically attacks a Victim."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['awarded', 'award', 'award', 'winning']

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
                    input_str += ' \n {}'.format('This is a Award event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['armed', 'unarmed', 'army', 'gunman', 'missile', 'clash']

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
                    input_str += ' \n {}'.format('This event is named Bearing_arms and its definition is "A Protagonist has a Weapon."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['became', 'become', 'becoming', 'turned', 'turning', 'formed']

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
                    input_str += ' \n {}'.format('This event is named Becoming and its definition is "an Entity ends up with some Final_quality."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['joined', 'join', 'qualified', 'joining', 'recruited', 'entered']

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
                    input_str += ' \n {}'.format('This event is named Becoming_a_member and its definition is "a New_member becomes a member of a socially constructed Group."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['operating', 'operated', 'operate', 'work', 'shut down', 'worked']

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
                    input_str += ' \n {}'.format('This event is named Being_in_operation and its definition is "a Device or machine is in service."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['siege', 'besieged', 'blockade', 'blockaded', 'besieging', 'Siege']

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
                    input_str += ' \n {}'.format('This event is named Besieging and its definition is "an Assailant surrounds the Victim to cut the Victim off from the outside."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['injured', 'wounded', 'suffered', 'injury', 'injuring', 'burned']

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
                    input_str += ' \n {}'.format('This event is named Bodily_harm and its definition is "a bodily injury to a Body_part."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['hit', 'lift', 'stretched', 'threw', 'bowl', 'caught']

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
                    input_str += ' \n {}'.format('This event is named Body_movement and its definition is "motions or actions an Agent performs using some part of body."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['breathing', 'breathe', 'breath']

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
                    input_str += ' \n {}'.format('This event is named Breathing and its definition is "an Agent causes Air to move in a direction."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['brought', 'bring', 'bringing', 'taken', 'led', 'carrying']

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
                    input_str += ' \n {}'.format('This event is named Bringing and its definition is "an Agent carries a Theme to some place."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['established', 'built', 'building', 'constructed', 'build', 'establish']

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
                    input_str += ' \n {}'.format('This event is named Building and its definition is "assembly or construction actions an Agent joins Components together to form a Created_entity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['carrying', 'carried', 'carry', 'Carried']

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
                    input_str += ' \n {}'.format('This event is named Carry_goods and its definition is "a Distributor sells lends or otherwise distributes a class of Goods."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['storm', 'hurricane', 'cyclone', 'crash', 'crashed', 'earthquake']

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
                    input_str += ' \n {}'.format('This event is named Catastrophe and its definition is "an Undesirable_event which affects the Patient negatively."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['caused', 'resulted in', 'led to', 'causing', 'due', 'force']

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
                    input_str += ' \n {}'.format('This event is named Causation and its definition is "a Cause causes an Effect."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['increased', 'increase', 'reduced', 'raised', 'peaked', 'rising']

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
                    input_str += ' \n {}'.format('This event is named Cause_change_of_position_on_a_scale and its definition is "an Agent or a Cause affects the position of an Item on some scale."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['weakened', 'intensified', 'strengthened', 'weakening', 'strengthening', 'reinforced']

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
                    input_str += ' \n {}'.format('This event is named Cause_change_of_strength and its definition is "an Agent causes a Patient to be more strong."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['combined', 'incorporated', 'merged', 'mixed', 'combination', 'united']

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
                    input_str += ' \n {}'.format('This event is named Cause_to_amalgamate and its definition is "Agent joining Parts to form a Whole."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['included', 'involved', 'include', 'consisted', 'involving', 'includes']

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
                    input_str += ' \n {}'.format('This event is named Cause_to_be_included and its definition is "An Agent or Cause makes a New_member part of Group."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['advanced', 'upgraded', 'promoted', 'advance', 'advancing', 'promote']

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
                    input_str += ' \n {}'.format('This event is named Cause_to_make_progress and its definition is "An Agent works on a Project so that it reaches a more advanced and desirable state."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['change', 'changed', 'transitioned', 'turned', 'degenerated', 'replaced']

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
                    input_str += ' \n {}'.format('This is a Change event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['cancelled', 'delayed', 'postponed', 'extended', 'canceled', 'delay']

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
                    input_str += ' \n {}'.format('This event is named Change_event_time and its definition is "an Agent or Cause changes the timing of an Event."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['revolution', 'election', 'uprising', 'replaced', 'coup', 'revolt']

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
                    input_str += ' \n {}'.format('This event is named Change_of_leadership and its definition is "the appointment of a New_leader or removal from office of an Old_leader."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['felt', 'feared', 'seemed', 'surprised', 'regarded', 'represented']

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
                    input_str += ' \n {}'.format('This is a Change_sentiment event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['changed', 'switch', 'transport', 'change']

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
                    input_str += ' \n {}'.format('This event is named Change_tool and its definition is "An Agent changes Tools from the Old_tool to a New_tool."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['confirmed', 'identified', 'classified', 'certified', 'confirm', 'identify']

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
                    input_str += ' \n {}'.format('This is a Check event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['election', 'elected', 'selected', 'chose', 'chosen', 'adopted']

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
                    input_str += ' \n {}'.format('This event is named Choosing and its definition is "A person decides upon the Chosen out of a set of Possibilities."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['joint', 'associated', 'allied', 'collaborated', 'association', 'associate']

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
                    input_str += ' \n {}'.format('This event is named Collaboration and its definition is "Partners work together in some Undertaking."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['met', 'gathered', 'assembled', 'meet', 'gathering', 'collected']

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
                    input_str += ' \n {}'.format('This event is named Come_together and its definition is "a group of Individuals meet to form a Configuration."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['occurred', 'formed', 'developed', 'emerged', 'originated', 'form']

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
                    input_str += ' \n {}'.format('This event is named Coming_to_be and its definition is "An Entity comes into existence at a particular Place and Time which may take a certain Duration_of_final_state."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['concluded', 'deemed', 'proved', 'found', 'thought', 'speculated']

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
                    input_str += ' \n {}'.format('This event is named Coming_to_believe and its definition is "A person comes to believe something."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['purchased', 'bought', 'purchase', 'buy', 'shopping', 'purchasing']

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
                    input_str += ' \n {}'.format('This event is named Commerce_buy and its definition is "a Buyer and a Seller exchanging Money and Goods."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['pay', 'paid', 'paying', 'payment', 'charged', 'refund']

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
                    input_str += ' \n {}'.format('This event is named Commerce_pay and its definition is "Buyers paying Money for Goods."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['sold', 'sale', 'selling', 'sell', 'marketed', 'auction']

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
                    input_str += ' \n {}'.format('This event is named Commerce_sell and its definition is "a seller exchanges money and goods with buyer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['committed', 'promised', 'vowed', 'pledged', 'commitment', 'guaranteed']

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
                    input_str += ' \n {}'.format('This event is named Commitment and its definition is "A Speaker makes a commitment to an Addressee to carry out some future action."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['crime', 'murder', 'committed', 'murdered', 'conspiracy', 'guilty']

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
                    input_str += ' \n {}'.format('This event is named Committing_crime and its definition is "A Perpetrator intentionally commits a Crime."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['said', 'negotiation', 'televised', 'debate', 'negotiated', 'negotiate']

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
                    input_str += ' \n {}'.format('This event is named Communication and its definition is "A Communicator conveys a Message to an Addressee."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['tournament', 'match', 'played', 'competition', 'race', 'game']

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
                    input_str += ' \n {}'.format('This event is named Competition and its definition is "people participate in an organized and rule governed activity ."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['faced', 'facing', 'suffered', 'challenge', 'confronted', 'failed']

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
                    input_str += ' \n {}'.format('This event is named Confronting_problem and its definition is "An Agent becomes involved in an Issue which has negative consequences for them."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['tied', 'linked', 'connected', 'contact', 'mounted', 'attached']

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
                    input_str += ' \n {}'.format('This event is named Connect and its definition is "affix a Connected_item or to bind onto a Fixed_location and is primarily so used."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['captured', 'defeated', 'capture', 'took', 'defeat', 'taken']

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
                    input_str += ' \n {}'.format('This event is named Conquering and its definition is "a Theme losing its autonomy and perhaps sustaining material damage as the result of a successful invasion on the behalf of a Conqueror."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['contained', 'contains', 'contain', 'consisting', 'comprised', 'comprises']

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
                    input_str += ' \n {}'.format('This event is named Containing and its definition is "a Container holds within its physical boundaries the Contents."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['control', 'command', 'occupied', 'commanded', 'controlled', 'ruled']

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
                    input_str += ' \n {}'.format('This event is named Control and its definition is "An Entity, Situation, or Variable control a Entity, Situation, or Variable."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['suggested', 'proved', 'recommended', 'convinced', 'advised', 'pressure']

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
                    input_str += ' \n {}'.format('This is a Convincing event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['cost', 'costing', 'expense', 'price', 'payment', 'free']

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
                    input_str += ' \n {}'.format('This is a Cost event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['filmed', 'designed', 'design', 'draw', 'depicted', 'drew']

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
                    input_str += ' \n {}'.format('This event is named Create_artwork and its definition is "A Creator creates an artifact that is typically an iconic Representation of an actual or imagined entity or event."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['produced', 'established', 'created', 'made', 'issued', 'founded']

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
                    input_str += ' \n {}'.format('This event is named Creating and its definition is "A Cause leads to the formation of a Created_entity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['investigation', 'trial', 'inquiry', 'investigated', 'indicted', 'investigate']

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
                    input_str += ' \n {}'.format('This event is named Criminal_investigation and its definition is "the inquiry and determination by an authority, the Investigator, of the circumstances surrounding an Incident perpetrated by a Suspect."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['treated', 'treatment', 'relieved', 'relief', 'relieve', 'relieving']

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
                    input_str += ' \n {}'.format('This event is named Cure and its definition is "a Healer treating and curing an Affliction of the Patient."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['damage', 'damaged', 'Damage', 'damaging', 'crashed', 'disrupted']

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
                    input_str += ' \n {}'.format('This event is named Damaging and its definition is "An Agent affects a Patient in such a way that the Patient ends up in a non canonical state."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['death', 'died', 'dead', 'casualty', 'fatality', 'perished']

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
                    input_str += ' \n {}'.format('This event is named Death and its definition is "the death of an Entity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['decided', 'determined', 'decision', 'determine', 'decide', 'considered']

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
                    input_str += ' \n {}'.format('This event is named Deciding and its definition is "A People makes a Decision."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['held', 'defense', 'defended', 'defending', 'defence', 'defensive']

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
                    input_str += ' \n {}'.format('This event is named Defending and its definition is "A Defender responds to an Assailant\'s attack on a Victim by directly and violently."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['left', 'leaving', 'leave', 'departed', 'departure', 'take off']

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
                    input_str += ' \n {}'.format('This event is named Departing and its definition is "An object moves away from a Source."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['destroyed', 'destruction', 'broke', 'destroying', 'destroy', 'collapsed']

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
                    input_str += ' \n {}'.format('This event is named Destroying and its definition is "A Destroyer or Entity affects the Patient negatively so that the Patient no longer exists.."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['dissipated', 'spread', 'dissipating', 'widespread', 'scattered', 'dissolved']

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
                    input_str += ' \n {}'.format('This event is named Dispersal and its definition is "An Agent or a Cause disperses or scatters Individuals from the Source."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['lost', 'loss', 'failed', 'losing', 'victory', 'win']

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
                    input_str += ' \n {}'.format('This event is named Earnings_and_losses and its definition is "An Earner receives Earnings by providing Goods to a Buyer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['training', 'trained', 'learned', 'studied', 'instruction', 'train']

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
                    input_str += ' \n {}'.format('This event is named Education_teaching and its definition is "A Student comes to learn either about a Subject Skill Precept or Fact as a result of instruction by a Teacher.."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['emergency']

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
                    input_str += ' \n {}'.format('This event is named Emergency and its definition is "someone should or must act to prevent a Undesirable_event from occurring."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['work', 'employed', 'hired', 'working', 'worked', 'employee']

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
                    input_str += ' \n {}'.format('This event is named Employment and its definition is "An Employer employs an Employee whose Position entails that the Employee perform certain Tasks in exchange for Compensation."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['evacuated', 'cleared', 'clear', 'evacuate', 'evacuation', 'cleanup']

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
                    input_str += ' \n {}'.format('This event is named Emptying and its definition is "empty containers and clear areas of some substance or items."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['fled', 'retreat', 'withdrew', 'escaped', 'escape', 'withdrawal']

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
                    input_str += ' \n {}'.format('This event is named Escaping and its definition is "A Self moving Escapee departs from an Undesirable_location."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['exchange', 'exchanged', 'trading', 'trade', 'changed', 'Exchange']

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
                    input_str += ' \n {}'.format('This event is named Exchange and its definition is "Two parties the Exchangers each give and receive from the other."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['expanded', 'extended', 'expansion', 'growing', 'expand', 'grew']

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
                    input_str += ' \n {}'.format('This event is named Expansion and its definition is "An Item changes its physical size."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['spent', 'exhausted', 'ran out of', 'depleted', 'spend', 'spending']

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
                    input_str += ' \n {}'.format('This event is named Expend_resource and its definition is "An Agent uses a Resource which is consumed and unavailable for future use."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['declared', 'announced', 'broadcast', 'declaration', 'aired', 'expressed']

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
                    input_str += ' \n {}'.format('This event is named Expressing_publicly and its definition is "A Communicator publicly communicates some difficult to express Content which they have had for some time."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['extradited', 'deported', 'extradition', 'bailed', 'indict', 'expel']

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
                    input_str += ' \n {}'.format('This event is named Extradition and its definition is "A Suspect in a Current_jurisdiction is forced by Authorities to go to the Crime_jurisdiction."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['filled', 'covered', 'covering', 'inundated', 'cover', 'flooded']

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
                    input_str += ' \n {}'.format('This event is named Filling and its definition is "fill containers and cover areas with some thing, things or substance."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['separated', 'married', 'marriage', 'consisted', 'separate', 'wedding']

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
                    input_str += ' \n {}'.format('This event is named Forming_relationships and its definition is "Partner interacts with another Partner to change their social relationship."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['prepared', 'prepare', 'preparing', 'ready', 'preparation', 'launched']

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
                    input_str += ' \n {}'.format('This is a GetReady event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['scored', 'win', 'gained', 'attained', 'winning', 'gain']

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
                    input_str += ' \n {}'.format('This event is named Getting and its definition is "A Recipient starts off without the Theme in their possession and then comes to possess it."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['abandoned', 'abandon', 'left', 'retired', 'pulled out', 'forgotten']

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
                    input_str += ' \n {}'.format('This is a GiveUp event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['given', 'gave', 'granted', 'give', 'offered', 'contributed']

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
                    input_str += ' \n {}'.format('This event is named Giving and its definition is "A Donor transfers a Theme from a Donor to a Recipient."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['access', 'cut off', 'available', 'shut down', 'accessible', 'link']

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
                    input_str += ' \n {}'.format('This event is named Having_or_lacking_access and its definition is "A Theme has access to a Useful_location or is blocked from it by a Barrier."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['cover', 'block', 'hidden', 'hiding', 'covering', 'blocked']

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
                    input_str += ' \n {}'.format('This event is named Hiding_objects and its definition is "An Agent causes a Object to become perceptually inaccessible to potential perceivers by placing it in a Hiding_place or putting in place an Obstruction that screens the object."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['disrupted', 'blocked', 'suppressed', 'hampered', 'interrupted', 'suppress']

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
                    input_str += ' \n {}'.format('This event is named Hindering and its definition is " Hindrance makes it more difficult for a Protagonist to complete their intended Action or a Hindrance."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['held', 'performed', 'took place', 'performing', 'perform', 'hold']

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
                    input_str += ' \n {}'.format('This is a Hold event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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


class Hostile_encounter(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['battle', 'fought', 'Battle', 'war', 'conflict', 'fighting']

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
                    input_str += ' \n {}'.format('This event is named Hostile_encounter and its definition is "opposing forces over a disputed Issue and/or in order to reach a specific Purpose."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['required', 'charged', 'charge', 'responsible', 'requires', 'levied']

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
                    input_str += ' \n {}'.format('This event is named Imposing_obligation and its definition is "A Duty is imposed on a Responsible_party according to a Principle which regulates how the Responsible_party should respond to a Situation."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['incident', 'accident', 'qualified', 'similarity', 'mistake', 'timing']

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
                    input_str += ' \n {}'.format('This is a Incident event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['affected', 'impact', 'effect', 'inspired', 'impacted', 'affect']

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
                    input_str += ' \n {}'.format('This is a Influence event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['absorbed', 'smoke', 'taken', 'drink', 'drunk', 'drank']

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
                    input_str += ' \n {}'.format('This event is named Ingestion and its definition is "An Ingestor consumes food or drink."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['hospitalized', 'formalize', 'formalised', 'institutionalized', 'decolonization', 'federalization']

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
                    input_str += ' \n {}'.format('This event is named Institutionalization and its definition is "A Patient is committed to the care of a medical Facility by a proper Authority."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['accused', 'blamed', 'condemned', 'charged', 'criticized', 'praised']

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
                    input_str += ' \n {}'.format('This event is named Judgment_communication and its definition is "A Communicator communicates a judgment of an Evaluee to an Addressee."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['justified', 'justify', 'justifying', 'recognize', 'reclaimed', 'explain']

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
                    input_str += ' \n {}'.format('This event is named Justifying and its definition is "An Agent uses some Means to explain why their Act or omission or a State_of_affairs they are involved in was licit despite appearances."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['kidnapping', 'kidnapped', 'hijacking', 'hijacked', 'abduction', 'hostage']

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
                    input_str += ' \n {}'.format('This event is named Kidnapping and its definition is "The words in this frame describe situations in which a Perpetrator carries off and holds the Victim against his or her will by force."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['killed', 'killing', 'massacre', 'murder', 'murdered', 'assassination']

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
                    input_str += ' \n {}'.format('This event is named Killing and its definition is "A Killer or Cause causes the death of the Victim."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['found', 'known', 'believed', 'discovered', 'noted', 'recognized']

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
                    input_str += ' \n {}'.format('This is a Know event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['brand', 'labeled', 'label', 'called', 'rebranded', 'noted']

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
                    input_str += ' \n {}'.format('This event is named Labeling and its definition is "A Speaker uses as a Label to refer to an Entity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['sentenced', 'convicted', 'ruled', 'sentence', 'verdict', 'acquitted']

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
                    input_str += ' \n {}'.format('This event is named Legal_rulings and its definition is "An Authority with the power to make decisions hands down a Finding over a question presented in a formal or informal Case."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['illegal', 'legal', 'lawsuit', 'prosecution', 'charged', 'convicted']

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
                    input_str += ' \n {}'.format('This event is named Legality and its definition is "an Action with respect to a Code of laws or rules."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['lighting', 'lightning', 'light', 'flared', 'flash', 'illuminated']

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
                    input_str += ' \n {}'.format('This is a Lighting event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['limited', 'limit', 'limiting', 'restricted', 'limitation', 'restriction']

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
                    input_str += ' \n {}'.format('This event is named Limiting and its definition is "An Agent or Cause limits a Range_of_options to having a certain Characteristic."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['made', 'produced', 'making', 'production', 'make', 'product']

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
                    input_str += ' \n {}'.format('This event is named Manufacturing and its definition is "A Producer produces a Product from a Resource for commercial purposes."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['operation', 'force', 'war', 'campaign', 'War', 'Operation']

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
                    input_str += ' \n {}'.format('This event is named Military_operation and its definition is "The military Force of a Possessor conducts large scale activities in a Area to accomplish military goals."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['moved', 'moving', 'flooding', 'followed', 'forced', 'turned']

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
                    input_str += ' \n {}'.format('This event is named Motion and its definition is "Some entity starts out in Source and ends up in Goal."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['passed', 'crossed', 'passing', 'fell', 'crossing', 'dropped']

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
                    input_str += ' \n {}'.format('This event is named Motion_directional and its definition is "a Theme moves in a certain Direction."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['named', 'called', 'referred', 'renamed', 'name', 'titled']

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
                    input_str += ' \n {}'.format('This event is named Name_conferral and its definition is "Speakers name Entities."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['open', 'opened', 'opening', 'close', 'closed', 'Open']

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
                    input_str += ' \n {}'.format('This event is named Openness and its definition is "A Useful_location is accessible to some Theme despite a potential Barrier."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['participated', 'played', 'attended', 'involved', 'participate', 'engaged']

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
                    input_str += ' \n {}'.format('This event is named Participation and its definition is "An Event with multiple Participants takes place."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['patrol', 'convoy', 'patrolling', 'escorted', 'guarding', 'escorting']

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
                    input_str += ' \n {}'.format('This event is named Patrolling and its definition is "An individual or group the Patrol moves through and examines a Ground in order to ensure that it is in a generally Desired_state_of_affairs."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['seen', 'observed', 'felt', 'viewed', 'see', 'watched']

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
                    input_str += ' \n {}'.format('This event is named Perception_active and its definition is "perceivers intentionally direct their attention to some entity or phenomenon in order to have a perceptual experience."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['set', 'placed', 'put', 'stationed', 'ranked', 'situated']

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
                    input_str += ' \n {}'.format('This event is named Placing and its definition is "an Agent places a Theme at Goal."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['practice', 'exercise', 'practiced', 'training', 'rehearsed', 'manoeuvred']

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
                    input_str += ' \n {}'.format('This event is named Practice and its definition is "An Agent enacts an Action that is intended to be performed again at one or more later Occasions."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['occurred', 'remained', 'present', 'presented', 'happened', 'appeared']

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
                    input_str += ' \n {}'.format('This event is named Presence and its definition is "An Entity exists at a particular Location, at a particular Time, as observed by an implicit observer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['maintained', 'retained', 'maintain', 'retain', 'retaining', 'survived']

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
                    input_str += ' \n {}'.format('This event is named Preserving and its definition is "an Agent preserves a Patient in order to prevent it from decaying."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['allowed', 'prevented', 'prevent', 'allowing', 'allow', 'avoid']

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
                    input_str += ' \n {}'.format('This event is named Preventing_or_letting and its definition is "A Potential_hindrance or an Agent keeps an Event from taking place."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['imprisonment', 'prisoner', 'prison', 'sentenced', 'sentence', 'jailed']

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
                    input_str += ' \n {}'.format('This event is named Prison and its definition is "Penal_institutions run by an Operator and may be intended to hold certain kinds of Inmates."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['ended', 'end', 'final', 'finished', 'ending', 'concluded']

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
                    input_str += ' \n {}'.format('This event is named Process_end and its definition is "A Process comes to an end."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['began', 'took place', 'started', 'launched', 'beginning', 'start']

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
                    input_str += ' \n {}'.format('This event is named Process_start and its definition is "An Event begins at a certain Time and Place."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['protest', 'demonstration', 'demonstrated', 'reason', 'appeal', 'Protests']

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
                    input_str += ' \n {}'.format('This event is named Protest and its definition is "A Protester expresses a strong opinion either in support of or against an Issue or Action."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['released', 'published', 'release', 'releasing', 'publication', 're', 'released']

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
                    input_str += ' \n {}'.format('This event is named Publishing and its definition is "A Publisher makes a Work of an Author available to some public Audience for general enjoyment, examination, and reference."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['disputed', 'dispute', 'debate', 'fought', 'argued', 'disagreement']

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
                    input_str += ' \n {}'.format('This event is named Quarreling and its definition is "A group of Arguers express incompatible opinions or beliefs about an Issue."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['ratified', 'treaty', 'authorized', 'endorsed', 'ratify', 'approval']

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
                    input_str += ' \n {}'.format('This event is named Ratification and its definition is "A Ratifier responds to a Proposal constructed by another party to make it binding over the Ratifier\'s jurisdiction or constituency."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['received', 'receive', 'receiving', 'accepted', 'accept', 'accepting']

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
                    input_str += ' \n {}'.format('This event is named Receiving and its definition is "A Recipient comes into possession of the Theme as a result of the joint action of the Donor and the Recipient."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['record', 'marked', 'recorded', 'recording', 'mark', 'documented']

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
                    input_str += ' \n {}'.format('This event is named Recording and its definition is "An Agent sets down in permanent form information about the occurrence of a Phenomenon."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['recovered', 'restored', 'regained', 'restore', 'recover', 'renewed']

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
                    input_str += ' \n {}'.format('This event is named Recovering and its definition is "the recovery or healing of a Patient from an Affliction without reference to the influence of any Treatment or Healer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['reform', 'set up', 'reformed', 'revolution', 'restructured', 'overhauled']

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
                    input_str += ' \n {}'.format('This event is named Reforming_a_system and its definition is "an Agent undertakes steps to change the structural makeup of a complex Entity with interdependent parts."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['released', 'release', 'freed', 'free', 'freedom', 'releasing']

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
                    input_str += ' \n {}'.format('This event is named Releasing and its definition is "A Captor ends the captivity or inhibition of the motion of a Theme from the Location_of_confinement."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['removed', 'eliminated', 'evacuated', 'expelled', 'cut', 'diverted']

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
                    input_str += ' \n {}'.format('This event is named Removing and its definition is "An Agent causes a Theme to move away from the Source."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['charter', 'chartered', 'renting', 'lease', 'rented', 'contracted']

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
                    input_str += ' \n {}'.format('This event is named Renting and its definition is "A Lessee gains the use of Goods owned by a Lessor by payment to the Lessor."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['reported', 'report', 'reporting', 'pointed out', 'point to', 'headline']

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
                    input_str += ' \n {}'.format('This event is named Reporting and its definition is "an Informer informs the Authorities of the illegal or otherwise improper Behavior of the Wrongdoer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['ordered', 'called', 'demanded', 'demand', 'asked', 'forced']

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
                    input_str += ' \n {}'.format('This event is named Request and its definition is "a Speaker asks an Addressee for something."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['rescue', 'rescued', 'saved', 'survived', 'save', 'saving']

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
                    input_str += ' \n {}'.format('This event is named Rescuing and its definition is "An Agent saves a Patient or an Asset from a Harmful_situation."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['test', 'research', 'study', 'investigate', 'investigated', 'investigating']

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
                    input_str += ' \n {}'.format('This event is named Research and its definition is "A Researcher attempts to answer a Question by means of consulting literature, observation, or conducting experiments in a particular Field pertinent to the Question."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['resolve', 'settled', 'deal', 'resolved', 'dealing', 'dealt']

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
                    input_str += ' \n {}'.format('This event is named Resolve_problem and its definition is "An Agent resolves an outstanding Problem by finding its solution explanation answer."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['response', 'responded', 'protest', 'action', 'respond', 'rejected']

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
                    input_str += ' \n {}'.format('This event is named Response and its definition is "An Agent performs a Response action in consequence of a Trigger event."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['revealed', 'confessed', 'exposed', 'disclosed', 'leaked', 'uncovered']

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
                    input_str += ' \n {}'.format('This event is named Reveal_secret and its definition is "A Speaker reveals Information that was previously secret to an Addressee."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['retaliation', 'revenge', 'retaliated', 'avenge', 'sanction', 'retaliate']

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
                    input_str += ' \n {}'.format('This event is named Revenge and its definition is "An Avenger performs a Punishment on a Offender as a consequence of an earlier action by the Offender."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['sentenced', 'executed', 'punish', 'execution', 'ticket', 'hanged']

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
                    input_str += ' \n {}'.format('This event is named Rewards_and_punishments and its definition is "An Agent performs a Response_action on an Evaluee for a Reason or actions or beliefs."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['risk', 'dared', 'dare', 'venture', 'venturing', 'fearing']

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
                    input_str += ' \n {}'.format('This event is named Risk and its definition is "A particular Situation is likely to result in a harmful event befalling an Asset."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['service', 'prayer', 'funeral', 'mourning', 'mass', 'burial']

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
                    input_str += ' \n {}'.format('This event is named Rite and its definition is "This frame concerns rituals performed in line with religious beliefs or tradition."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['looting', 'looted', 'plundered', 'robbery', 'robbed', 'ransacked']

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
                    input_str += ' \n {}'.format('This event is named Robbery and its definition is "a Perpetrator wrongs a Victim by taking something from them."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['swept', 'search', 'sweeping', 'manhunt', 'searched', 'searching']

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
                    input_str += ' \n {}'.format('This event is named Scouring and its definition is "A Searcher looks all over a Ground in order to find a Sought_entity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['investigation', 'search', 'reconnaissance', 'monitor', 'census', 'surveillance']

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
                    input_str += ' \n {}'.format('This event is named Scrutiny and its definition is "a person paying close attention to something in order to discover and note its salient characteristics."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['marched', 'landfall', 'landing', 'sailed', 'tracked', 'followed']

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
                    input_str += ' \n {}'.format('This event is named Self_motion and its definition is "The Self_mover moves under its own direction along a Path."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['sent', 'send', 'delivered', 'sending', 'dispatched', 'transport']

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
                    input_str += ' \n {}'.format('This event is named Sending and its definition is "A Sender plans the Path of a Theme and places it in circumstances such that it travels along this Path."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['signed', 'signing', 'treaty', 'signed on', 'agreement', 'sign']

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
                    input_str += ' \n {}'.format('This event is named Sign_agreement and its definition is "A Signatory signs an Agreement document and takes on a commitment encoded in the Agreement."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['festival', 'concert', 'event', 'show', 'hosted', 'Festival']

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
                    input_str += ' \n {}'.format('This event is named Social_event and its definition is "A Social_event occurs at which Attendees are present to conduct a social function or joint activity."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['claimed', 'stated', 'said', 'described', 'claim', 'state']

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
                    input_str += ' \n {}'.format('This event is named Statement and its definition is "a Speaker addresses a Message to some Addressee using language."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['filed', 'submitted', 'submit', 'petition', 'bankruptcy', 'finalized']

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
                    input_str += ' \n {}'.format('This event is named Submitting_documents and its definition is "A Submittor gives Documents to an Authority so that they can be processed as part of an application."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['supply', 'service', 'provided', 'provide', 'sponsored', 'providing']

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
                    input_str += ' \n {}'.format('This event is named Supply and its definition is "A Supplier gives a Theme to a Recipient to fulfill a need or purpose of the Recipient."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['support', 'supported', 'supporting', 'supporter', 'bolstered', 'Support']

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
                    input_str += ' \n {}'.format('This event is named Supporting and its definition is "A Supporter assists to strengthen the Supported by lending itself in material aid."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['surrendered', 'surrender', 'ceded', 'capitulated', 'surrendering', 'cede']

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
                    input_str += ' \n {}'.format('This event is named Surrendering and its definition is "a Fugitive presents himself or herself to the Authorities to be subject to the criminal process."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['surrounded', 'surrounding', 'encompassed', 'surround', 'area', 'Surrounded']

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
                    input_str += ' \n {}'.format('This event is named Surrounding and its definition is "A Figure surrounds a Ground."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['suspected', 'suspect']

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
                    input_str += ' \n {}'.format('This event is named Suspicion and its definition is "An Authority believes that the Suspect is a participant in a criminal Incident."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['said', 'told', 'informed', 'talk', 'say', 'telling']

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
                    input_str += ' \n {}'.format('This event is named Telling and its definition is "A Speaker addresses an Addressee with a Message."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['remained', 'stayed', 'kept', 'remaining', 'keeping', 'stay']

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
                    input_str += ' \n {}'.format('This event is named Temporary_stay and its definition is "A Guest stays for a time at a Location."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['bombing', 'terrorist', 'attack', 'terrorism', 'suicide attack', 'bomb']

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
                    input_str += ' \n {}'.format('This event is named Terrorism and its definition is "A Terrorist commits a violent or otherwise harmful Act upon a Victim in order to coerce or terrorize a government or populace."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['test', 'testing', 'tested', 'trial', 'assessment', 'examined']

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
                    input_str += ' \n {}'.format('This is a Testing event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['stolen', 'stole', 'hacked', 'stealing', 'looted', 'plundered']

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
                    input_str += ' \n {}'.format('This event is named Theft and its definition is "a Perpetrator takes Goods from a Victim or a Source."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['tour', 'Tour', 'visited', 'toured', 'travelling', 'visit']

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
                    input_str += ' \n {}'.format('This event is named Traveling and its definition is "a Traveler moves from a Source location to a Goal along a Path or within an Area."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['shot', 'shooting', 'fired', 'fire', 'shot down', 'explosion']

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
                    input_str += ' \n {}'.format('This event is named Use_firearm and its definition is "An Agent causes a Firearm to discharge."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['used', 'use', 'using', 'applied', 'us', 'utilized']

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
                    input_str += ' \n {}'.format('This event is named Using and its definition is "An Agent manipulates an Instrument in order to achieve a Purpose."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['violence', 'riot', 'conflict', 'unrest', 'violent', 'uprising']

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
                    input_str += ' \n {}'.format('This event is named Violence and its definition is "an Aggressor or Cause injuring a Victim."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['sang', 'cry', 'shouting', 'singing', 'song', 'yelled']

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
                    input_str += ' \n {}'.format('This event is named Vocalizations and its definition is "sounds produced by animate entities by means of their vocal tracts."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['warning', 'threatened', 'threat', 'warned', 'alerted', 'alert']

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
                    input_str += ' \n {}'.format('This event is named Warning and its definition is "A Speaker warns an Addressee by giving a Message that describes an undesirable situation that may affect the Addressee."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['wore', 'wearing', 'dressed', 'wear', 'dress', 'disguised']

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
                    input_str += ' \n {}'.format('This event is named Wearing and its definition is "Clothing the Wearer has on."')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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
        return ['wrote', 'written', 'signed', 'writing', 'listed', 'write']

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
                    input_str += ' \n {}'.format('This is a Writing event type.')
                if i_style == 'keywords':
                    input_str += ' \n Possible triggers include {}'.format(', '.join(self.get_keywords()))
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

