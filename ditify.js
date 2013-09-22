var ditify = function(options){

	options = options || {}
	this.attribs = options.attribs || []
	this.label = options.label || options.attribs[options.attribs.length-1] 
	this.data = []
	this.data_info = []
	this.trees = []

	this.Node = function(){
	    this.label = "";
	    this.attribute = "";
	    this.isLeaf = true;
	    this.chance = 1;
	    this.children = []
	}

	this.count_distinct_values = function(table, attrib)
	{
	    var distinct = {};
	    for(var i=0; i<table.length; i++){
	        if(table[i][table] in distinct)
	            distinct[table[i][table]]++;
	        else
	            distinct[table[i][table]] = 1;
	    }
	    return distinct;
	}

	this.get_split_attrib = function(table)
	{
	    var min_entropy = Infinity, min_entropy_attrib = "";

	    for(attrib in this.attribs){
	    	if(attrib == this.label) continue;

	    	var distinct_values = this.count_distinct_values(table, attrib);
	    	var attrib_entropy = 0;

	    	for(value in distinct_values){
	    		var pruned_table = this.prune(table, attrib, value);
	    		var distinct_labels = this.count_distinct_values(pruned_table, this.label);

	    		var label_entropy = 0;
	    		for(_label in distinct_labels){
	    			var label_chance = distinct_labels[_label]/pruned_table.length;
	    			label_entropy -= (p * (Math.log(p)/Math.log(2)));
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
	}

	this.count_frequent_values = function(table, attrib)
	{
	    var distinct_values = count_distinct_values(table, attrib);
	    var max_frequency = -1, max_value = "", entropy = 0;

	    for(value in distinct_values){
	        if (distinct_values[value] > max_frequency){
				max_frequency = distinct_values[value];
	            max_value = value;
	        }
	        entropy -= (distinct_values[value]/table.length) * Math.log(distinct_values[value]/table.length)/Math.log(2);
	    }
	    return {value: max_value, chance: max_frequency/table.length, entropy: entropy, distinct_count: Object.keys(distinct).length};
	}

	this.is_homogeneous = function(table, label_attrib)
	{
	    var start_value = table[0][label_attrib];
	    for(var i = 1; i < table.length; i++){
	        if(start_value != table[i][label_attrib])
	            return false;
	    }
	    return true;		
	}


	this.generate_tree = function(node, table, default_label)
	{
	    var tree = {leaves:0, entropy:0};

	    if(table.length < 1 || Object.keys(table[0]).length == 1){

	        node.attribute = default_label.value;
	        node.chance = default_label.chance;

	        return { leaves: default_label.distinct_count, entropy: default_label.entropy };
	    }
	    else if(is_homogeneous(table, this.label)){

	        node.attribute = table[0][this.label];
	        return { leaves: 1, entropy: 0 };        
	    }
	    else
	    {
	        var split_attrib = get_split_attrib(table);
	        node.attribute = split_attrib;
	        node.isLeaf = false;

	        for(value in this.data_info[split_attrib]){
	            var child_node = new Node();

	            child_node.label = value;
	            node.children.push(child_node);

	            var child_default_label = count_frequent_values(table, this.label);
	            var child_table = this.prune(table, split_attrib, value);
	            var child_tree = 
	            	generate_tree(child_node, child_table, child_default_label);

	            tree.entropy += child_tree.entropy;
	            tree.leaves += child_tree.leaves;
	        }            
	    }
	    return tree;
	}
}

ditify.prototype = {

	prune: function(table, attrib, value){
		var pruned = [];
		var _table = jQuery.extend(true, {}, table);

	    for(var i = 0; i < _table.length; i++){
	        if(_table[i][attrib] == value){
	            delete _table[i][column];
	            pruned.push(_table[i]);
	        }
	    }
	    return pruned;   
	}

	train: function(values){
		if((values.length || Object.keys(values).length) < this.attribs.length)
			throw "Incomplete training data";

		if(typeof values == "array"){
			var _data = {};
			for(var i=0; i<values.length; i++){
				_data[this.attribs[i]] = values[i];
			}
			this.data.push(_data);
		}
		else if(typeof values == "object"){
			this.data.push(values);
		}
		else{
			throw "Invalid training data";
		}
	}
}