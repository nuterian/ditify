isObject = function(obj){ return obj === Object(obj) }
isArray = function(obj){ return Object.prototype.toString.call(obj) == '[object Array]' }
clone = function(obj) { return (JSON.parse(JSON.stringify(obj))) }

var ditify = function(options){

	options = options || {}
	this.attribs = options.attribs || []
	this.label = options.label || this.attribs[this.attribs.length-1] 
	this.data = []
	this.data_info = {}
	this.tree = null

	for(var i = 0; i < this.attribs.length; i++){ this.data_info[this.attribs[i]] = {} }

}

ditify.prototype = {

	Node: function(){
	    this.label = "";
	    this.attribute = "";
	    this.isLeaf = true;
	    this.chance = 1;
	    this.children = []
	},

	count_distinct_values: function(table, attrib)
	{
	    var distinct = {};
	    for(var i=0; i<table.length; i++){
	        if(table[i][attrib] in distinct)
	            distinct[table[i][attrib]]++;
	        else
	            distinct[table[i][attrib]] = 1;
	    }

	    return distinct;
	},

	get_split_attrib: function(table)
	{
	    var min_entropy = Infinity, min_entropy_attrib = "";

	    for(attrib in table[0]){
	    	if(attrib == this.label) continue;

	    	var distinct_values = this.count_distinct_values(table, attrib);
	    	
	    	var attrib_entropy = 0;

	    	for(value in distinct_values){
	    		var pruned_table = this.prune(table, attrib, value);
	    		var distinct_labels = this.count_distinct_values(pruned_table, this.label);

	    		var label_entropy = 0;
	    		for(_label in distinct_labels){
	    			var label_chance = distinct_labels[_label]/pruned_table.length;
	    			label_entropy -= (label_chance * (Math.log(label_chance)/Math.log(2)));
	    		}
	    		attrib_entropy += (attrib_entropy * distinct_values[value]);
	    	}
	    	attrib_entropy /= table.length;

	        if(attrib_entropy < min_entropy){
	            min_entropy = attrib_entropy;
	            min_entropy_attrib = attrib;
	        }
	    }

	    return min_entropy_attrib;
	},

	count_frequent_values: function(table, attrib)
	{
	    var distinct_values = this.count_distinct_values(table, attrib);
	    var max_frequency = -1, max_value = "", entropy = 0;

	    for(value in distinct_values){
	        if (distinct_values[value] > max_frequency){
				max_frequency = distinct_values[value];
	            max_value = value;
	        }
	        entropy -= (distinct_values[value]/table.length) * Math.log(distinct_values[value]/table.length)/Math.log(2);
	    }
	    return {value: max_value, chance: max_frequency/table.length, entropy: entropy, distinct_count: Object.keys(distinct_values).length};
	},

	is_homogeneous: function(table, label_attrib)
	{
	    var start_value = table[0][label_attrib];
	    for(var i = 1; i < table.length; i++){
	        if(start_value != table[i][label_attrib])
	            return false;
	    }
	    return true;		
	},


	generate_tree: function(node, table, default_label)
	{
	    var tree = {leaves:0, entropy:0};

	    if(table.length < 1 || Object.keys(table[0]).length == 1){

	        node.attribute = default_label.value;
	        node.chance = default_label.chance;

	        return { leaves: default_label.distinct_count, entropy: default_label.entropy };
	    }
	    else if(this.is_homogeneous(table, this.label)){

	        node.attribute = table[0][this.label];
	        return { leaves: 1, entropy: 0 };        
	    }
	    else
	    {
	        var split_attrib = this.get_split_attrib(table);
	        node.attribute = split_attrib;
	        node.isLeaf = false;

	        for(split_value in this.data_info[split_attrib]){
	            var child_node = new this.Node();

	            child_node.label = split_value;
	            node.children.push(child_node);

	            var child_default_label = this.count_frequent_values(table, this.label);
	            var child_table = this.prune(table, split_attrib, split_value);

	            var child_tree = 
	            	this.generate_tree(child_node, child_table, child_default_label);

	            tree.entropy += child_tree.entropy;
	            tree.leaves += child_tree.leaves;
	        }            
	    }
	    return tree;
	},

	find_label: function(node, test_data){
		if(node.isLeaf) return { label: node.attribute, chance: node.chance }
		for(var i=0; i<node.children.length; i++){
			if(test_data[node.attribute] == node.children[i].label)
				return this.find_label(node.children[i], test_data);
		}
	},


	prune: function(table, attrib, value){
		var pruned = [];
		var _table = clone(table);

	    for(var i = 0; i < _table.length; i++){
	        if(_table[i][attrib] == value){
	            delete _table[i][attrib];
	            pruned.push(_table[i]);
	        }
	    }
	    return pruned;   
	},

	train: function(values){
		if((values.length || Object.keys(values).length) < this.attribs.length)
			throw "Incomplete training data.";

		if(isArray(values)){
			var _data = {};
			for(var i=0; i<values.length; i++){
				_data[this.attribs[i]] = values[i];

				if(values[i] in this.data_info[this.attribs[i]])
					this.data_info[this.attribs[i]][values[i]]++;
				else
					this.data_info[this.attribs[i]][values[i]] = 1;
			}
			this.data.push(_data);
		}
		else if(isObject(values)){
			this.data.push(values);
			for(value in values){
				if(values[value] in this.data_info[value])
					this.data_info[value][values[value]]++;
				else
					this.data_info[value][values[value]] = 1;
			}
		}
		else{
			throw "Invalid training data.";
		}
		this.tree = null
	},

	classify: function(values){
		if((values.length || Object.keys(values).length) < this.attribs.length)
			throw "Incomplete training data.";	
		
		var label_attrib = "", test_data = {}
		if(isArray(values)){
			for(var i=0; i<values.length; i++){
				if(values[i] == "") label_attrib = this.attribs[i];
				else test_data[this.attribs[i]] = values[i];
			}

		}
		else if(isObject(values)){
			for(var i=0; i<this.attribs.length; i++){
				if(not (this.attribs[i] in values)){
					label_attrib = this.attribs[i];
					break;
				}
			}
			test_data = values;
		}
		else{
			throw "Invalid test data.";
		}

		if(this.label != label_attrib) this.setLabel(label_attrib);

		if(this.tree == null){
			var node = new this.Node();
			var def_label = this.count_frequent_values(this.data, this.label);

			this.tree = this.generate_tree(node, this.data, def_label);
			this.tree["root"] = node;
		}

		return this.find_label(this.tree.root, test_data)
	},

	setLabel: function(attrib){
		this.label = attrib
		this.tree = null
	}


}