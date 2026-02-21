import copy
from enum import Enum


class Table:
    """Represents a table with foreign key and primary key relationships"""

    def __init__(self, table_name, primary_key=["id"], table_nn_attribute=None, table_size=1000, csv_file_location=None,
                 attributes=None, irrelevant_attributes=None, keep_fk_attributes=None, sample_rate=1.0, fd_list=None,
                 no_compression=None, condition = None):
    # def __init__(self, table_name, primary_key=["id"], table_nn_attribute=None, table_size=1000, csv_file_location=None,
    #              attributes=None, irrelevant_attributes=None, keep_fk_attributes=None, sample_rate=1.0,
    #              no_compression=None):

        self.table_name = table_name
        self.table_size = table_size
        self.primary_key = primary_key

        ###
        # condition
        self.condition  = condition

        self.csv_file_location = csv_file_location
        self.attributes = attributes
        self.irrelevant_attributes = irrelevant_attributes
        if irrelevant_attributes is None:
            self.irrelevant_attributes = []
        self.keep_fk_attributes = keep_fk_attributes
        #keep_fk_attributes 是一个用于指定在数据表中保留外键属性的属性列表。
        #如果给定keep_fk_attributes，那么相应的外键属性就不会被视为无关属性而被删除。
        if keep_fk_attributes is None:
            self.keep_fk_attributes = []
        self.no_compression = no_compression
        if no_compression is None:
            self.no_compression = []

        if fd_list is None:
            self.fd_list = []
        else:
            self.fd_list = [(table_name + '.' + fd_source, table_name + '.' + fd_dest) for fd_source, fd_dest in
                            fd_list]
        ###

        # additional attribute indicating whether tuple is NULL (can occur since we learn SPN on FULL OUTER JOIN)
        if table_nn_attribute is None:
            self.table_nn_attribute = self.table_name + '_nn'

        # FK references
        self.outgoing_relationships = []

        # referenced as FK
        self.incoming_relationships = []
        self.sample_rate = sample_rate

    # "fd = functional dependence ?"
    def children_fd_attributes(self, attribute):
        return [fd_source for fd_source, fd_dest in self.fd_list if fd_dest == attribute]

    def parent_fd_attributes(self, attribute):
        return [fd_dest for fd_source, fd_dest in self.fd_list if fd_source == attribute]
    ###


class Relationship:
    """Foreign key primary key relationship"""

    def __init__(self, start, end, start_attr, end_attr, multiplier_attribute_name):
        self.start = start.table_name
        self.start_attr = start_attr

        self.end = end.table_name
        self.end_attr = end_attr

        ###
        #rdc score
        self.rdc_score = None
        self.correlation_level = None
        self.max_rdc_attributes = None

        # matching tuples in FULL OUTER JOIN
        # the name of the attribute that represents the number of matching tuples in
        # a FULL OUTER JOIN between the two tables
        self.multiplier_attribute_name = multiplier_attribute_name

        # The multiplier_attribute_name parameter represents the number of matching tuples,
        # which includes both null and non-null tuples. However,
        # in some cases, we may only be interested in the number of non-null matching tuples.
        # This is where the multiplier_attribute_name_nn parameter comes in.
        # matching tuples (not NULL)
        self.multiplier_attribute_name_nn = multiplier_attribute_name + '_nn'

        self.identifier = self.start + '.' + self.start_attr + \
                          ' = ' + self.end + '.' + self.end_attr

        # for start table we are outgoing relationship
        start.outgoing_relationships.append(self)
        end.incoming_relationships.append(self)


class SchemaGraph:
    """Holds all tables and relationships"""

    def __init__(self):
        self.tables = []
        self.relationships = []
        self.table_dictionary = {}
        self.relationship_dictionary = {}

        ###
        # adjacent list for join graph
        self.adjacent_list = []
        self.adjacent_list_dictionary = {}

        ###
        # strongly correlated、 weakly correlated and medium correlated tables in a relationship
        self.strongly_correlated_relationship = []
        self.strongly_correlated_relationship_dictionary = {}

        self.medium_correlated_relationship = []
        self.medium_correlated_relationship_dictionary = {}

        self.weakly_correlated_relationship = []
        self.weakly_correlated_relationship_dictionary = {}

    def add_table(self, table):
        self.tables.append(table)
        self.table_dictionary[table.table_name] = table

    def find_relationship_two(self, start_table, end_table, relationship_list):
        for relationship_obj in relationship_list:
            if relationship_obj.start_table == start_table and relationship_obj.end_table == end_table :
                return  relationship_obj
            else:
                return None
    def add_relationship(self, start_name, start_attr, end_name, end_attr, multiplier_attribute_name=None):
        if multiplier_attribute_name is None:
            multiplier_attribute_name = 'mul_' + start_name + '.' + start_attr

        relationship = Relationship(self.table_dictionary[start_name],
                                    self.table_dictionary[end_name],
                                    start_attr,
                                    end_attr,
                                    multiplier_attribute_name)

        self.relationships.append(relationship)
        self.relationship_dictionary[relationship.identifier] = relationship
        ###
        # add adjacent list for the join graph
        if relationship.start not in self.adjacent_list:
            self.adjacent_list.append(relationship.start)
            self.adjacent_list_dictionary[relationship.start] = set()
            self.adjacent_list_dictionary[relationship.start].add(relationship.end)

            if relationship.end not in self.adjacent_list:
                self.adjacent_list.append(relationship.end)
                self.adjacent_list_dictionary[relationship.end] = set()
                self.adjacent_list_dictionary[relationship.end].add(relationship.start)
            else:
                self.adjacent_list_dictionary[relationship.end].add(relationship.start)
        else:
            self.adjacent_list_dictionary[relationship.start].add(relationship.end)

            if relationship.end not in self.adjacent_list:
                self.adjacent_list.append(relationship.end)
                self.adjacent_list_dictionary[relationship.end] = set()
                self.adjacent_list_dictionary[relationship.end].add(relationship.start)
            else:
                self.adjacent_list_dictionary[relationship.end].add(relationship.start)

        return relationship.identifier


class QueryType(Enum):
    AQP = 0
    CARDINALITY = 1


class AggregationType(Enum):
    SUM = 0
    AVG = 1
    COUNT = 2

###
#correlation level of two tables in a relationship
class CorrelationLevel(Enum):
    Weak = 0     # rdc_score < 0.3
    Medium = 1   # 0.3 < rdc_score < 0.7
    Strong = 2   # rdc_score > 0.7


class AggregationOperationType(Enum):
    PLUS = 0
    MINUS = 1
    AGGREGATION = 2


class Query:
    """Represents query"""

    def __init__(self, schema_graph, query_type=QueryType.CARDINALITY, features=None):
        self.query_type = query_type
        self.schema_graph = schema_graph
        self.table_set = set()
        self.relationship_set = set()
        self.table_where_condition_dict = {}
        self.conditions = []
        self.aggregation_operations = []
        self.group_bys = []

    def remove_conditions_for_attributes(self, table, attributes):
        def conflicting(condition):
            return any([condition.startswith(attribute + ' ') or condition.startswith(attribute + '<') or
                        condition.startswith(attribute + '>') or condition.startswith(attribute + '=') for
                        attribute in attributes])

        if self.table_where_condition_dict.get(table) is not None:
            self.table_where_condition_dict[table] = [condition for condition in
                                                      self.table_where_condition_dict[table]
                                                      if not conflicting(condition)]
        self.conditions = [(cond_table, condition) for cond_table, condition in self.conditions
                           if not (cond_table == table and conflicting(condition))]

    def copy_cardinality_query(self):
        query = Query(self.schema_graph)
        query.table_set = copy.copy(self.table_set)
        query.relationship_set = copy.copy(self.relationship_set)
        query.table_where_condition_dict = copy.copy(self.table_where_condition_dict)
        query.conditions = copy.copy(self.conditions)
        return query

    def add_group_by(self, table, attribute):
        self.group_bys.append((table, attribute))

    def add_aggregation_operation(self, operation):
        """
        Adds operation to AQP query.
        :param operation: (AggregationOperationType.AGGREGATION, operation_type, operation_factors) or (AggregationOperationType.MINUS, None, None)
        :return:
        """
        self.aggregation_operations.append(operation)

    def add_join_condition(self, relationship_identifier):

        relationship = self.schema_graph.relationship_dictionary[relationship_identifier]
        self.table_set.add(relationship.start)
        self.table_set.add(relationship.end)

        self.relationship_set.add(relationship_identifier)

    def add_where_condition(self, table, condition):
        if self.table_where_condition_dict.get(table) is None:
            self.table_where_condition_dict[table] = [condition]
        else:
            self.table_where_condition_dict[table].append(condition)
        self.conditions.append((table, condition))
