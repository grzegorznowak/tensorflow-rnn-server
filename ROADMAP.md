1. Create a server that simulates random responses from [0; 100] range using normal distribution
2. Update server to build and load variables of a specific graph using TF's `saver.restore(sess, restore_name)`
3. Make it reset RNN's state on every day (how do we know what a new day is ? Possibly using timestamp data ?)
4. Make it aware of what a data batch is (5 by default, configurable later on ?) so that it can update RNN's state every full batch
5. Start sending a proper RNN output in response to each request  