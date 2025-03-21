import numpy as np
from typing import Optional, Union, List, Callable
from mytorch.utils import GradientBuffer


class Operation:
    def __init__(
        self,
        inputs: List[np.ndarray],
        output: np.ndarray,
        gradients_to_update: List[Optional[Union[np.ndarray, None]]],
        backward_operation: Callable,
    ):
        """
        Args:
            - inputs: operation inputs (List[np.ndarray])
            - outputs: operation output (Optional[Union[np.ndarray, List[np.ndarray]]])
            - gradients_to_update: parameter gradients if for parameter of ,
                        network or None (numpy.ndarray, None)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Note: You do not need to modify anything here
        """
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation

        self.i0_shp = self.inputs[0].shape
        self.i1_shp = None
        if len(self.inputs) > 1:
            self.i1_shp = self.inputs[1].shape
        self.bwd_op_name = self.backward_operation.__name__

    def __repr__(self):
        """
        Use this with print(operation) to help debug.
        """
        return f"Operation [{self.i0_shp}, {self.i1_shp}, {self.output.shape}, {self.gradients_to_update}, {self.bwd_op_name}]"


class Autograd:
    def __init__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        A check to make sure you don't create more than 1 Autograd at a time. You can remove
        this if you want to do multiple in parallel. We do not recommend this
        """
        if getattr(self.__class__, "_has_instance", False):
            raise RuntimeError("Cannot create more than 1 Autograd instance")
        self.__class__._has_instance = True

        self.gradient_buffer = GradientBuffer()
        self.operation_list = []

    def __del__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        Class destructor. We use this for testing purposes.
        """
        del self.gradient_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(
        self,
        inputs: List[np.ndarray],
        output: np.ndarray,
        gradients_to_update: List[Optional[Union[np.ndarray, None]]],
        backward_operation: Callable,
    ):
        """
        Adds operation to operation list and puts gradients in gradient buffer for tracking
        Args:
            - inputs: operation inputs (numpy.ndarray)
            - outputs: operation output (numpy.ndarray)
            - gradients_to_update: parameter gradients if for parameter of
                        network or None (numpy.ndarray, None)
                NOTE: Given the linear layer as shown in the writeup section
                    2.4 there are 2 kinds of inputs to an operation:
                    1) one that requires gradients to be internally tracked
                        ex. input (X) to a layer
                    2) one that requires gradient to be externally tracked
                        ex. weight matrix (W) of a layer (so we can track dW)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Returns:
            No return required
        """
        if len(inputs) != len(gradients_to_update):
            raise Exception(
                "Number of inputs must match the number of gradients to update!"
            )

        # TODO: Add all of the inputs to the self.gradient_buffer using the add_spot() function
        # This will allow the gradients to be tracked
        for input in inputs:
            self.gradient_buffer.add_spot(input)
        # ??
        # Also add the output to the gradient buffer
        self.gradient_buffer.add_spot(output)

        # TODO: Append an Operation object to the self.operation_list
        operation = Operation(
            inputs=inputs,
            output=output,
            gradients_to_update=gradients_to_update,
            backward_operation=backward_operation
        )
        self.operation_list.append(operation)

    # def backward(self, divergence):
    #     """
    #     The backpropagation through the self.operation_list with a given divergence.
    #     This function should automatically update gradients of parameters by checking
    #     the gradients_to_update. Read the write up for further explanation
    #     Args:
    #         - divergence: loss value (float/double/int/long)
    #     Returns:
    #         No return required
    #     """
    #     # TODO: Iterate through the self.operation_list and propagate the gradients.
    #     # NOTE: Make sure you iterate in the correct direction. How are gradients propagated?
    #     for i, operation in enumerate(reversed(self.operation_list)):
    #         # TODO: For the first iteration set the gradient to be propagated equal to the divergence.
    #         # For the remaining iterations the gradient to be propagated can be retrieved from the
    #         # self.gradient_buffer.get_param.
    #         if i == 0: #the first iter
    #             grad = divergence
    #         else:
    #             grad = self.gradient_buffer.get_param(operation.output)

    #         # TODO: Execute the backward for the Operation
    #         # NOTE: Make sure to unpack the inputs list if you aren't parsing a list in your backward.
    #         grad_all_inputs = operation.backward_operation(grad, *operation.inputs)
        
    #         """
    #         ?? not sure do i need this?
    #         # If backward_operation returns a single gradient (not a tuple or list),
    #         # convert it to a list for consistent handling
    #         """
    #         if not isinstance(grad_all_inputs, (list, tuple)):
    #             grad_all_inputs = [grad_all_inputs]

    #         # TODO: Loop through the inputs and their corresponding gradients.
    #         for j, (input, grad_to_update) in enumerate(zip(operation.inputs, operation.gradients_to_update)):
    #             grad = grad_all_inputs[j]

    #             # Check with the Operation's gradients_to_update if you need to
    #             if grad is not None:
    #                 if grad_to_update is not None:
    #                     operation.gradients_to_update[j] += grad
    #                 # directly update a gradient, and do the following accordingly:
    #                 else:
    #                     #   1) Inputs with internally tracked gradients: update the gradient stored in
    #                     #   self.gradient_buffer
    #                     current_grad = self.gradient_buffer.get_param(input)
    #                     if current_grad is None:
    #                         self.gradient_buffer.update_param(input, grad)
    #                     #   2) Inputs with externally tracked gradients: update gradients_to_update
    #                     # NOTE: Make sure the order of gradients align with the order of inputs
    #                     else:
    #                         # print("grad_test",grad)
    #                         self.gradient_buffer.update_param(input, current_grad + grad)
    def backward(self, divergence):
        for i, operation in enumerate(reversed(self.operation_list)):
            # 获取当前操作的梯度
            if i == 0:  # 第一次迭代
                grad = divergence
            else:
                grad = self.gradient_buffer.get_param(operation.output)

            # 执行反向操作获取所有输入的梯度
            grad_all_inputs = operation.backward_operation(grad, *operation.inputs)
            
            # 如果返回单个梯度而非列表，将其转换为列表
            if not isinstance(grad_all_inputs, (list, tuple)):
                grad_all_inputs = [grad_all_inputs]
                
            # 处理每个输入的梯度
            for j, (input, grad_to_update) in enumerate(zip(operation.inputs, operation.gradients_to_update)):
                # 获取当前输入的梯度
                input_grad = grad_all_inputs[j]
                
                # 确保梯度不是None
                if input_grad is not None:
                    # 检查是否需要直接更新梯度
                    if grad_to_update is not None:
                        # 外部跟踪梯度（如权重）
                        operation.gradients_to_update[j] += input_grad
                    else:
                        # 内部跟踪梯度（如层输入）
                        current_grad = self.gradient_buffer.get_param(input)
                        if current_grad is None:
                            self.gradient_buffer.update_param(input, input_grad)
                        else:
                            self.gradient_buffer.update_param(input, current_grad + input_grad)

    def zero_grad(self):
        """
        Resets gradient buffer and operations list. No need to modify.
        """
        self.gradient_buffer.clear()
        self.operation_list = []
