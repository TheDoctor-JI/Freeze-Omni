import time
from collections import deque
import heapq

class ContextSerializer:
    """
    Serializes audio features from user and system based on timestamps
    to handle overlapping speech scenarios.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the serializer state"""
        ## Track actual IPU state of the user
        self.user_in_actual_ipu = False

        ## The pseudo IPU state may not be consistent with the actual IPU state of the system. It simply reflects whether we have been sending system features to the dialog state predictor or not. The reason we need to track this is to enable the setting of chat template prefix on the first system feature in a pseudo IPU sequence.
        self.system_in_pseudo_ipu = False 


        ## Priority queue for timestamp-based ordering
        ## Each item is (timestamp, identity, feature_data, status)
        self.feature_queue = []


        print("ContextSerializer state has been reset.")
    
    def add_feature_chunk(self, feature_chunk):
        """
        Add a feature chunk to the priority queue.
        
        Parameters:
        - feature_chunk: a dictionary containing: {'identity', 'feature', 'status', 'time_stamp'}
        """
        timestamp = feature_chunk.get('time_stamp', None)
        identity = feature_chunk.get('identity', None)
        status = feature_chunk.get('status', None)
        feature = feature_chunk.get('feature', None)

        
        # Add to priority queue (heapq uses min-heap, so earliest timestamp first)
        heapq.heappush(self.feature_queue, (timestamp, identity, status, feature))
    
    def gate_feature(self, identity, status):
        """
        Check whether a feature should be sent to the dialog stat predictor based on its identity and status.
        
        Parameters:
        - feature_data (dict): Feature chunk data
        - identity (str): 'user' or 'system'
        
        Returns:
        - to_send: True if the feature should be sent, False otherwise
        - force_ipu_sl: True if the status should be forced to 'ipu_sl' for the first feature in a sequence to set the chat template prefix
        """

        to_send = False
        force_ipu_sl = False 
        
        if identity == 'user':
            ## User feature always gets sent to the dialog state predictor
            to_send = True
            force_ipu_sl = False ## Since we always respect the user IPU structure, we do not need to force the status to ipu_sl. Human chunks can just prreserve their status as they are. The chat template will be automatically set based on the status of the first feature in a sequence.

            ## Track the actual IPU state of the user
            if status == 'ipu_sl' or status == 'ipu_cl':
                self.user_in_actual_ipu = True
            elif status == 'ipu_el':
                self.user_in_actual_ipu = False

            ## Entering user IPU means resetting the pseudo IPU state of the system
            self.system_in_pseudo_ipu = False
                        
        elif identity == 'system':
            if self.user_in_actual_ipu:
                ## No system feature should be sent if user is in actual IPU
                pass

            else:
                ## If user is not in actual IPU, we can send system feature
                to_send = True

                ## Check if this one is the first system feature in a sequence
                if not self.system_in_pseudo_ipu:
                    self.system_in_pseudo_ipu = True
                    force_ipu_sl = True  # Force status to ipu_sl for the first system feature in a pseudo IPU sequence to facilitate the setting of chat template prefix
        
        return to_send, force_ipu_sl

    def get_next_feature(self):
        """
        Get the next feature to process based on timestamp ordering and a serialization strategy.
        
        Returns:
        - feature_to_send: a feature chunk to be processed next by the dialog state predictor, or None
        """        

        feature_to_send = None

        if not self.feature_queue:
            # No features to process
            return feature_to_send

        ## Pop the earliest feature from the priority queue
        timestamp, identity, status, feature_data = heapq.heappop(self.feature_queue)
        
        to_send, force_ipu_sl = self.gate_feature(identity, status)

        if to_send:
            feature_to_send = {
                'time_stamp': timestamp,
                'identity': identity,
                'status': status if not force_ipu_sl else 'ipu_sl',  # Force status to ipu_sl if necessary
                'feature': feature_data,
            }
     
        return feature_to_send
