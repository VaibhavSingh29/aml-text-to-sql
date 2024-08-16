import json
import anytree
from anytree import Node
from model.utils.smbop import ra_postproc, ra_preproc
from model.utils.smbop import moz_sql_parser as msp

POST_ORDER_LIST = []

class SmBopSQL2RA:
    def __init__(self) -> None:
        pass
    
    def generate_ast(self, sql_query):
        try: 
            tree_dict = msp.parse(sql_query)
            return tree_dict['query']
        except msp.ParseException as e:
            print(f'Could not create AST for: {sql_query}')
            return None

    def ast_to_ra(self, query_ast):
        if not query_ast:
            return None
        tree_object = ra_preproc.ast_to_ra(query_ast)
        arit_list = anytree.search.findall(
                tree_object, filter_=lambda x: x.name in ["sub", "add"]
            )
        haslist_list = anytree.search.findall(
                    tree_object,
                    filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
                )
        if arit_list or haslist_list:
            print(f'Could not create RA for:  {query_ast}')
            return None
        else:
            return tree_object
    
    def post_order(self, node):
        if  len(node.children) == 0:
            POST_ORDER_LIST.append('(')
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')
        elif len(node.children) == 1:
            POST_ORDER_LIST.append('(')
            self.post_order(node.children[0] if node.children[0] else node.children[1])
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')
        else:
            POST_ORDER_LIST.append('(')
            self.post_order(node.children[0])
            self.post_order(node.children[1])
            try:
                name = str(node.name) + '=' + str(node.val)
            except:
                name = str(node.name)
            POST_ORDER_LIST.append(name)
            POST_ORDER_LIST.append(')')


if __name__ == '__main__':
    with open('../spider/train_spider.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        sql_query = item['query'] 
        print('----------------------------------------------------------------------------------------------')
        print(sql_query)
        smbop = SmBopSQL2RA()
        tree_object = smbop.ast_to_ra(smbop.generate_ast(sql_query=sql_query))
        if not tree_object:
            continue
        POST_ORDER_LIST = []
        smbop.post_order(tree_object)
        print(''.join(POST_ORDER_LIST))


