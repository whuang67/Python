# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:47:59 2017

@author: whuang67
"""

class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None
        
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
        
    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element
            
    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
        counter = 1
        current = self.head
        if position < 1:
            return None
        while current and counter <= position:
            if counter == position:
                return current
            current = current.next
            counter += 1
        return None
    
    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""
        counter = 1
        current = self.head
        if position > 1:
            while current and counter < position:
                if counter == position - 1:
                    new_element.next = current.next
                    current.next = new_element
                current = current.next
                counter += 1
        elif position == 1:
            new_element.next = current.next
            self.head = new_element

    
    
    def delete(self, value):
        """Delete the first node with a given value."""
        current = self.head
        previous = None
        while current.value != value and current.next:
            previous = current
            current = current.next
        if current.value == value:
            if previous:
                previous.next = current.next
            else:
                self.head = current.next

# Test cases
# Set up some Elements
e1 = Element(1)
e2 = Element(2)
e3 = Element(3)
e4 = Element(4)

# Start setting up a LinkedList
ll = LinkedList(e1)
ll.append(e2)
ll.append(e3)

# Test get_position
# Should print 3
print ll.head.next.next.value
# Should also print 3
print ll.get_position(3).value

# Test insert
ll.insert(e4,3)
# Should print 4 now
print ll.get_position(3).value

# Test delete
ll.delete(1)
# Should print 2 now
print ll.get_position(1).value
# Should print 4 now
print ll.get_position(2).value
# Should print 3 now
print ll.get_position(3).value








class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None
        
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
        
    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def insert_first(self, new_element):
        new_element.next = self.head
        self.head = new_element

    def delete_first(self):
        "Delete the first (head) element in the LinkedList as return it"
        if self.head:
            delete_element = self.head
            temp = delete_element.next
            self.head = temp
            return delete_element
        else:
            return None

class Stack(object):
    def __init__(self,top=None):
        self.ll = LinkedList(top)

    def push(self, new_element):
        "Push (add) a new element onto the top of the stack"
        self.ll.insert_first(new_element)

    def pop(self):
        "Pop (remove) the first element off the top of the stack and return it"
        return self.ll.delete_first()
    
# Test cases
# Set up some Elements
e1 = Element(1)
e2 = Element(2)
e3 = Element(3)
e4 = Element(4)

# Start setting up a Stack
stack = Stack(e1)

# Test stack functionality
stack.push(e2)
stack.push(e3)
print stack.pop().value
print stack.pop().value
print stack.pop().value
print stack.pop()
stack.push(e4)
print stack.pop().value







class Queue:
    def __init__(self, head=None):
        self.storage = [head]

    def enqueue(self, new_element):
        self.storage.append(new_element)

    def peek(self):
        return self.storage[0]

    def dequeue(self):
        return self.storage.pop[0]
    
# Setup
q = Queue(1)
q.enqueue(2)
q.enqueue(3)

# Test peek
# Should be 1
print q.peek()

# Test dequeue
# Should be 1
print q.dequeue()

# Test enqueue
q.enqueue(4)
# Should be 2
print q.dequeue()
# Should be 3
print q.dequeue()
# Should be 4
print q.dequeue()
q.enqueue(5)
# Should be 5
print q.peek()
len([1,2,3])




import numpy as np

def binary_search(input_array, value):
    """Your code goes here."""
    input_array = np.array(input_array)
    Result = True
    out = -1
    array = input_array
    position = 0
    while Result:
        cutoff = len(array)/2
        position = position + cutoff
        if value > array[cutoff]:
            array = array[cutoff:]
            position = position + len(array)/2
        elif value < array[cutoff]:
            array = array[:cutoff]
            position = position - len(array)/2
            
        else:
            out = position-2
            Result = False
        
        if len(array) == 1:
            Result = False
    
    return out

test_list = [1,3,9,11,15,19,29]
test_val1 = 25
test_val2 = 19
print binary_search(test_list, test_val1)
print binary_search(test_list, test_val2)







test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
position = len(test)-1

while position >= 0:
    for i in range(0, len(test[:position])):
        if test[i] > test[position]:
            temp = test[i]
            test[i] = test[position]
            test[position] = temp
        print test
    position -= 1

    
print test
temp = test[0]
print test
print temp
test[0] = test[9]
print test
print temp
test[9] = temp
print test
print temp


ord("a")
chr(97)

class HashTable(object):
    def __init__(self):
        self.table = [None]*10000

    def store(self, string):
        if len(string) <= 10000:
            self.table = list(string)

    def lookup(self, string):
        if self.table[0] != None:
            hash_value = ord(self.table[0])*100 + ord(self.table[1])
        else:
            hash_value = -1
        return hash_value

    def calculate_hash_value(self, string):
        string_list = list(string)
        hash_value = ord(string_list[0])*100 + ord(string_list[1])
        return hash_value
    
# Setup
hash_table = HashTable()

# Test calculate_hash_value
# Should be 8568
print hash_table.calculate_hash_value('UDACITY')
# Test lookup edge case
# Should be -1
print hash_table.lookup('UDACITY')
# Test store
hash_table.store('UDACITY')
# Should be 8568
print hash_table.lookup('UDACITY')

# Test store edge case
hash_table.store('UDACIOUS')
# Should be 8568
print hash_table.lookup('UDACIOUS')




class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

    def search(self, find_val):
        return self.preorder_search(self.root, find_val)

    def print_tree(self):
        return self.preorder_print(self.root, "")[:-1]

    def preorder_search(self, start, find_val):
        if start:
            if start.value == find_val:
                return True
            else:
                return self.preorder_search(start.left, find_val) or \
                       self.preorder_search(start.right, find_val)
        return False

    def preorder_print(self, start, traversal):
        if start:
            traversal += (str(start.value) + '-')
            traversal = self.preorder_print(start.left, traversal)
            traversal = self.preorder_print(start.right, traversal)
        return traversal


# Set up tree
tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)

# Test search
# Should be True
print tree.search(4)
# Should be False
print tree.search(6)

# Test print_tree
# Should be 1-2-4-5-3
print tree.print_tree()
